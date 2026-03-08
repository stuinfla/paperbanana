"""
SVG Visualizer Agent — Generates publication-quality SVG diagrams via code generation.

Instead of asking an image model to draw pixels, this agent asks an LLM to write SVG code.
Result: 100% text accuracy, infinite scalability, editable output.

Includes a self-correcting render loop:
  1. LLM generates SVG code from the styled description
  2. SVG is rendered to PNG for evaluation
  3. Critic evaluates the rendered PNG
  4. If improvements needed, LLM edits the SVG code
  5. Repeat until quality threshold met (max 3 iterations)
"""

import asyncio
import base64
import io
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure Cairo library is findable on macOS (Homebrew install)
_brew_prefix = subprocess.run(['brew', '--prefix', 'cairo'], capture_output=True, text=True).stdout.strip()
if _brew_prefix and os.path.isdir(f"{_brew_prefix}/lib"):
    os.environ.setdefault("DYLD_LIBRARY_PATH", f"{_brew_prefix}/lib")

from PIL import Image
from google.genai import types

from utils import generation_utils
from .base_agent import BaseAgent

# Tags that may appear in planner/stylist output
_HIERARCHY_TAG_RE = re.compile(r'\[(?:PRIMARY|SECONDARY|TERTIARY)\]\s*')


def _render_svg_to_png(svg_code: str, width: int = 2400, height: int = 1600) -> Optional[str]:
    """
    Render SVG to PNG and return base64-encoded JPEG.
    Tries cairosvg first, falls back to Pillow SVG support, then rsvg-convert.
    """
    try:
        import cairosvg
        png_data = cairosvg.svg2png(
            bytestring=svg_code.encode('utf-8'),
            output_width=width,
            output_height=height,
        )
        img = Image.open(io.BytesIO(png_data)).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except ImportError:
        pass
    except Exception as e:
        print(f"[SVG Visualizer] cairosvg render failed: {e}")

    # Fallback: rsvg-convert (often available on macOS via librsvg)
    try:
        with tempfile.NamedTemporaryFile(suffix='.svg', mode='w', delete=False) as f:
            f.write(svg_code)
            svg_path = f.name

        png_path = svg_path.replace('.svg', '.png')
        result = subprocess.run(
            ['rsvg-convert', '-w', str(width), '-h', str(height), '-o', png_path, svg_path],
            capture_output=True, timeout=30
        )

        if result.returncode == 0 and Path(png_path).exists():
            img = Image.open(png_path).convert('RGB')
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=95)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[SVG Visualizer] rsvg-convert fallback failed: {e}")
    except Exception as e:
        print(f"[SVG Visualizer] rsvg-convert error: {e}")

    # Last resort: write SVG and convert with Pillow (limited SVG support)
    print("[SVG Visualizer] Warning: No SVG renderer available. Returning SVG as-is without PNG preview.")
    return None


def _extract_svg_from_response(text: str) -> str:
    """Extract SVG code from LLM response, handling markdown code blocks."""
    # Try to extract from code block first
    svg_match = re.search(r'```(?:svg|xml)?\s*\n(.*?)\n```', text, re.DOTALL)
    if svg_match:
        return svg_match.group(1).strip()

    # Try to find raw SVG
    svg_match = re.search(r'(<svg[\s\S]*?</svg>)', text, re.DOTALL)
    if svg_match:
        return svg_match.group(1).strip()

    # Return the whole thing if it looks like SVG
    if '<svg' in text:
        return text.strip()

    return text.strip()


class SVGVisualizerAgent(BaseAgent):
    """
    Generates SVG diagrams from text descriptions using LLM code generation.
    Produces 100% accurate text, scalable vector output.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.model_name  # Use reasoning model, NOT image model
        self.system_prompt = SVG_VISUALIZER_SYSTEM_PROMPT
        self.max_fix_iterations = 2  # render → evaluate → fix cycles

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SVG from the styled description, then run vision-based critic loop.
        The critic SEES the rendered PNG and provides specific spatial/design fixes
        that get applied surgically to the SVG code.
        """
        task_name = "diagram"

        # Find the best available description (prefer stylist, fall back to planner)
        desc_key = None
        for key in [
            f"target_{task_name}_stylist_desc0",
            f"target_{task_name}_desc0",
        ]:
            if key in data:
                desc_key = key
                break

        if not desc_key:
            print("[SVG Visualizer] No description found in data. Skipping.")
            return data

        # Clean the description
        description = data[desc_key]
        clean_desc = _HIERARCHY_TAG_RE.sub('', description)
        clean_desc = re.sub(r'Element count:\s*\d+/\d+', '', clean_desc).strip()

        # Get visual intent for additional context
        visual_intent = data.get('visual_intent', '')

        # Generate initial SVG
        prompt = SVG_GENERATION_PROMPT.format(
            description=clean_desc,
            visual_intent=visual_intent,
        )

        content_list = [{"type": "text", "text": prompt}]

        response_list = await generation_utils.call_gemini_with_retry_async(
            model_name=self.model_name,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.7,
                candidate_count=1,
                max_output_tokens=50000,
            ),
            max_attempts=3,
            retry_delay=5,
        )

        if not response_list or response_list[0] == "Error":
            print("[SVG Visualizer] Failed to generate SVG code.")
            return data

        svg_code = _extract_svg_from_response(response_list[0])
        data[f"{desc_key}_svg_code"] = svg_code

        # Render to PNG
        rendered = _render_svg_to_png(svg_code)
        if rendered:
            data[f"{desc_key}_base64_jpg"] = rendered
            print(f"[SVG Visualizer] Generated and rendered SVG for {desc_key}")
        else:
            print(f"[SVG Visualizer] Generated SVG but could not render to PNG")
            return data

        # === VISION-BASED CRITIC LOOP ===
        # The critic SEES the rendered PNG and provides specific fixes
        max_critic_rounds = getattr(self.exp_config, 'max_critic_rounds', self.max_fix_iterations)
        if max_critic_rounds <= 0:
            return data

        svg_key = f"{desc_key}_svg_code"
        img_key = f"{desc_key}_base64_jpg"

        for round_idx in range(max_critic_rounds):
            current_rendered = data.get(img_key)
            if not current_rendered or len(current_rendered) < 100:
                break

            # Send rendered PNG to vision critic
            print(f"[SVG Visualizer] Vision critic round {round_idx + 1}/{max_critic_rounds}...")
            critique = await self._vision_critique(current_rendered, clean_desc, visual_intent)

            if not critique:
                print(f"[SVG Visualizer] Critic returned no feedback, stopping.")
                break

            # Check if critic says it's good enough
            if "NO_CHANGES_NEEDED" in critique:
                print(f"[SVG Visualizer] Critic approved diagram (round {round_idx + 1}).")
                break

            print(f"[SVG Visualizer] Critic feedback: {critique[:200]}...")

            # Apply fixes to SVG
            data = await self.fix_svg(data, critique, svg_key)
            print(f"[SVG Visualizer] Applied fixes, round {round_idx + 1} complete.")

        return data

    async def _vision_critique(self, rendered_base64: str, description: str, visual_intent: str) -> Optional[str]:
        """
        Send the rendered PNG to a multimodal model for visual critique.
        Returns specific, actionable fix instructions or 'NO_CHANGES_NEEDED'.
        """
        critique_prompt = VISION_CRITIC_PROMPT.format(
            description=description,
            visual_intent=visual_intent,
        )

        content_list = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": rendered_base64,
                },
            },
            {"type": "text", "text": critique_prompt},
        ]

        try:
            response_list = await generation_utils.call_gemini_with_retry_async(
                model_name=self.model_name,
                contents=content_list,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert visual design critic. You evaluate rendered SVG diagrams for quality issues.",
                    temperature=0.3,
                    candidate_count=1,
                    max_output_tokens=4000,
                ),
                max_attempts=2,
                retry_delay=5,
            )

            if response_list and response_list[0] != "Error":
                return response_list[0]
        except Exception as e:
            print(f"[SVG Visualizer] Vision critique failed: {e}")

        return None

    async def fix_svg(self, data: Dict[str, Any], critique: str, svg_key: str) -> Dict[str, Any]:
        """
        Fix SVG based on critic feedback. This is the self-correcting loop.
        """
        current_svg = data.get(svg_key, '')
        if not current_svg:
            return data

        prompt = SVG_FIX_PROMPT.format(
            current_svg=current_svg,
            critique=critique,
        )

        content_list = [{"type": "text", "text": prompt}]

        response_list = await generation_utils.call_gemini_with_retry_async(
            model_name=self.model_name,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.4,  # Even lower temp for fixes
                candidate_count=1,
                max_output_tokens=50000,
            ),
            max_attempts=3,
            retry_delay=5,
        )

        if response_list and response_list[0] != "Error":
            fixed_svg = _extract_svg_from_response(response_list[0])
            data[svg_key] = fixed_svg

            rendered = _render_svg_to_png(fixed_svg)
            if rendered:
                # Update the image key
                image_key = svg_key.replace('_svg_code', '_base64_jpg')
                data[image_key] = rendered
                print(f"[SVG Visualizer] Fixed and re-rendered SVG")

        return data


# ============================================================================
# PROMPTS
# ============================================================================

SVG_VISUALIZER_SYSTEM_PROMPT = """You are an expert SVG diagram designer who creates publication-quality EXPLANATORY technical diagrams.

Your diagrams EXPLAIN complex concepts. They are NOT abstract art. They are NOT decorative illustrations.
A good diagram is like a great teacher: it uses visuals AND words together to make someone understand something in 15 seconds that would take 5 minutes to read.

MANDATORY RULES:
1. Output ONLY valid SVG code wrapped in ```svg code blocks. No explanation text outside the SVG.
2. All text MUST be perfectly spelled and legible. Use font-size 13-28px. Never smaller than 12px.
3. Use system-ui, -apple-system, sans-serif as the font family.
4. Canvas size: viewBox="0 0 1200 800", width="1200", height="800".
5. Use a cohesive color palette of 3-5 colors. Define gradients in <defs>.
6. The most important concept must be the largest, most visually prominent element.
7. Generous whitespace between elements. Never crowd the canvas.
8. Maximum 20 distinct visual elements. Clarity > simplicity > complexity.
9. Use rounded rectangles (rx="8-12"), clean lines, subtle drop shadows.
10. NO figure title/caption at the top — but EVERY element must have explanatory text.

INFORMATION DENSITY — THIS IS CRITICAL:
- Every visual element MUST have a label AND a short description (1 line, 5-15 words)
- Labels alone are NOT enough. "Identity" means nothing. "Identity: Each agent has a unique profile with trust scores and expertise domains" — THAT explains.
- Use 2-level text: bold/large label (16-22px) + smaller explanation underneath (13-15px, lighter color)
- Arrows and connections MUST be labeled to show what flows between elements
- Include a 1-2 sentence summary text block that states the core insight
- If someone sees this diagram and doesn't understand the concept better, the diagram FAILED

DESIGN PHILOSOPHY:
- The diagram must TEACH, not just illustrate
- Use spatial relationships AND text together — neither alone is sufficient
- The viewer should understand the core concept AND its key details in 15 seconds
- Use color FUNCTIONALLY (to encode meaning), not decoratively
- Shapes should represent real concepts (containers, flows, layers), not abstract decoration
- Think: "Would this work as a slide in a presentation to someone who knows nothing about this topic?"

SVG TECHNIQUES TO USE:
- <defs> for reusable gradients, filters, markers
- <g> groups for logical element grouping with transform
- <filter> for subtle drop shadows (feDropShadow)
- Clean arrows with <marker> for directional flow
- <text> with proper anchoring — use separate <text> elements for each line
- Rounded <rect> containers to group related concepts
- Subtle opacity variations to create visual hierarchy (primary=1.0, secondary=0.85, tertiary=0.7)

CAIRO RENDERING RULES (MUST FOLLOW — these cause real visual bugs):
- NEVER use <tspan> with different fill/color on the same <text> line — Cairo renders them overlapping. Use SEPARATE <text> elements with different y positions instead.
- NEVER use emoji characters (💡🔥⚡ etc.) — Cairo renders them as empty squares. Use SVG shapes instead.
- NEVER use unicode arrows (→ ← ↑ ↓) in text — they render as squares. Use SVG <path> or <line> with <marker> arrowheads.
- Keep 20px minimum vertical spacing between text lines to prevent overlap.
- Place arrow labels in clear space — never behind or overlapping with boxes.
"""

SVG_GENERATION_PROMPT = """Create an EXPLANATORY SVG diagram based on this description:

{description}

Visual intent: {visual_intent}

CRITICAL REQUIREMENTS — READ CAREFULLY:
This diagram must EXPLAIN the concept, not just illustrate it. Someone with zero context should look at this and understand:
1. What this system/concept IS (core purpose in 1 sentence)
2. What the key components ARE and what each one DOES (not just names — functions)
3. How the components RELATE to each other (flows, dependencies, layers)

EVERY element must have:
- A bold label (16-22px, white/bright)
- A 1-line description underneath (13-15px, lighter gray like #94a3b8) explaining what it does
- Example: Don't just write "Consensus" — write "Consensus" with "Agents debate, vote, and converge on verified truth" below it

LAYOUT:
- viewBox="0 0 1200 800"
- Dark background (#0B1426 or similar deep navy) with light text (#e2e8f0)
- Warm accent colors (amber #F59E0B, orange #F97316) for primary elements
- Cool colors (blue #3B82F6, indigo #6366F1) for secondary elements
- Gray (#64748b) for tertiary/supporting text
- Use rounded rectangles as containers with subtle borders
- Clean arrows between components showing data/control flow
- Label the arrows too (e.g., "knowledge flows", "trust scores", "CSI data")
- Include a prominent 1-2 sentence "core insight" text block somewhere on the canvas

ANTI-PATTERNS — DO NOT DO THESE:
- Do NOT create abstract shapes without explanatory text
- Do NOT use single-word labels without descriptions
- Do NOT make "artistic" diagrams that look pretty but explain nothing
- Do NOT overlap text with other elements
- Do NOT use font sizes below 12px
- Do NOT create purely decorative elements that don't represent real concepts

Output ONLY the SVG code in a ```svg code block."""

SVG_FIX_PROMPT = """Fix this SVG diagram based on the following critique:

CRITIQUE:
{critique}

CURRENT SVG:
```svg
{current_svg}
```

Apply the requested changes. Maintain the overall design and color palette.

IMPORTANT CAIRO RENDERING RULES (these cause real bugs if ignored):
- Do NOT use <tspan> with different fill colors on the same <text> line — Cairo renders them overlapping. Use separate <text> elements instead.
- Do NOT use emoji characters — Cairo renders them as empty squares. Use SVG shapes (circles, icons) instead.
- Do NOT use unicode arrows (→, ←) in text — use SVG <path> or <line> with markers instead.
- Keep all text elements well-separated — at least 20px vertical spacing between text lines.
- Arrow labels should never overlap with boxes or other text. Place them in clear space.

Output ONLY the fixed SVG code in a ```svg code block."""

VISION_CRITIC_PROMPT = """Look at this rendered SVG diagram carefully. The diagram was supposed to explain:

CONCEPT: {description}

INTENT: {visual_intent}

Evaluate the RENDERED image (not the code) for these specific issues:

1. TEXT PROBLEMS: Is any text overlapping, clipped, cut off, or unreadable? Are there empty squares (failed emoji/unicode rendering)? Is any text too small to read?

2. LAYOUT PROBLEMS: Are elements crowded or overlapping? Is there wasted whitespace? Are arrows pointing to the wrong places? Are labels misaligned with their elements?

3. INFORMATION GAPS: Does every element have both a label AND a description? Are arrow connections labeled? Is there a core insight summary? Would someone unfamiliar with the topic understand this in 15 seconds?

4. DESIGN ISSUES: Is the visual hierarchy clear (most important = largest/boldest)? Is the color scheme consistent? Do the visual containers make logical sense?

If the diagram scores 95/100 or higher with NO text rendering issues, respond with exactly: NO_CHANGES_NEEDED

Otherwise, provide SPECIFIC, ACTIONABLE fixes. For each issue:
- Describe exactly WHAT is wrong (e.g., "The text 'raw knowledge' at the bottom-left is partially hidden behind the Collective box")
- Describe exactly HOW to fix it (e.g., "Move the 'raw knowledge' label 30px higher so it sits above the arrow, not behind the box")
- Be spatial and precise — reference positions (top-left, center, bottom-right), approximate coordinates, and specific text strings

Do NOT provide vague feedback like "improve the layout." Every fix must be specific enough to translate directly into SVG coordinate changes."""
