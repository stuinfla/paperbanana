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
        Generate SVG from the styled description.
        Stores both SVG code and rendered PNG (base64 JPEG) in data.
        """
        task_name = "diagram"  # SVG visualizer only handles diagrams

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

        # Generate SVG
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
                temperature=0.7,  # Lower temp for code generation
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

SVG_VISUALIZER_SYSTEM_PROMPT = """You are an expert SVG diagram designer who creates publication-quality technical diagrams.

You generate clean, well-structured SVG code that renders as compelling visual explanations.

MANDATORY RULES:
1. Output ONLY valid SVG code wrapped in ```svg code blocks. No explanation text.
2. All text MUST be perfectly spelled and legible. Use font-size 12-28px. Never smaller than 11px.
3. Use system-ui, -apple-system, sans-serif as the font family.
4. Canvas size: viewBox="0 0 1200 800", width="1200", height="800".
5. Use a cohesive color palette of 3-5 colors. Define gradients in <defs>.
6. The most important concept must be the largest, most visually prominent element.
7. Generous whitespace between elements. Never crowd the canvas.
8. Maximum 15 distinct visual elements. Simplicity > complexity.
9. Use rounded rectangles (rx="8-12"), clean lines, subtle drop shadows.
10. NO figure titles or captions — the diagram speaks for itself.

DESIGN PHILOSOPHY:
- The diagram must tell a STORY, not list components
- Use spatial relationships to show how concepts connect
- The viewer should understand the core idea in 5 seconds
- Use color FUNCTIONALLY (to encode meaning), not decoratively
- Whitespace is a structural element, not wasted space

SVG TECHNIQUES TO USE:
- <defs> for reusable gradients, filters, markers
- <g> groups for logical element grouping with transform
- <filter> for subtle drop shadows (feDropShadow)
- Curved <path> elements for organic flow connections
- <text> with proper anchoring and font styling
- Subtle opacity variations to create depth
"""

SVG_GENERATION_PROMPT = """Create an SVG diagram based on this description:

{description}

Visual intent: {visual_intent}

Generate a complete, valid SVG that visualizes this concept as a compelling diagram.
The diagram should make someone instantly understand the concept — not through labels and boxes,
but through a visual metaphor that clicks.

Remember:
- viewBox="0 0 1200 800"
- Dark background (#0B1426 or similar deep navy) with light text
- Warm accent colors (amber #F59E0B, orange #F97316) for primary elements
- Cool colors (blue #3B82F6, purple #6366F1) for secondary elements
- Maximum 15 elements
- Text must be crisp and correctly spelled

Output ONLY the SVG code in a ```svg code block."""

SVG_FIX_PROMPT = """Fix this SVG diagram based on the following critique:

CRITIQUE:
{critique}

CURRENT SVG:
```svg
{current_svg}
```

Apply the requested changes. Maintain the overall design and color palette.
Output ONLY the fixed SVG code in a ```svg code block."""
