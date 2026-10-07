"""
app.py - AURA EDIT Gradio Web Application.
Clean, minimal, professional interface for generating 6-second vertical Aura Edits.
Strictly plain-text UI with no emoji or decorative unicode characters.
"""

import os
import sys
import time
import tempfile
import logging
import traceback
import gradio as gr

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.frames import extract_clip_frames
from pipeline.face import detect_face_and_zoom_target, get_best_face_crop
from pipeline.emotion import analyze_emotion
from pipeline.compose import compose_aura_video

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("aura_edit.app")

SAMPLE_CLIP_PATH = os.path.join(PROJECT_ROOT, "demo_samples", "sample_portrait.mp4")

CUSTOM_CSS = """
/* AURA EDIT Minimal Dark Theme */
:root {
    --accent: #e11d48;
    --accent-hover: #be123c;
    --accent-focus: rgba(225, 29, 72, 0.35);
    --bg-page: #09090b;
    --bg-card: #121216;
    --bg-input: #18181d;
    --border-subtle: #23232b;
    --border-input: #272732;
    --border-hover: #3f3f4c;
    --text-primary: #f4f4f5;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --text-danger: #f43f5e;
}

body, .gradio-container {
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.aura-container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 16px 48px;
}

/* Header */
.aura-header {
    text-align: center;
    margin-bottom: 32px;
}

.aura-title {
    font-size: 32px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 6px 0;
}

.aura-subtitle {
    font-size: 15px;
    font-weight: 400;
    color: var(--text-secondary);
    margin: 0;
}

/* Layout Columns */
.main-row {
    gap: 24px !important;
    align-items: flex-start !important;
}

.right-column {
    position: sticky !important;
    top: 24px !important;
}

/* Cards */
.aura-card {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
}

.card-title {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    text-transform: uppercase;
}

.card-helper {
    font-size: 13px;
    color: var(--text-muted);
    margin: 8px 0 0 0;
    line-height: 1.4;
}

/* Compact Video Input */
.compact-video .wrap,
.compact-video video,
.compact-video .upload-container {
    max-height: 280px !important;
    border-radius: 10px !important;
}

/* Buttons */
.secondary-btn {
    flex: 1 1 0% !important;
    width: 100% !important;
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-input) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    cursor: pointer !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
}

.secondary-btn:hover {
    background-color: var(--border-subtle) !important;
    border-color: var(--border-hover) !important;
}

.primary-btn {
    width: 100% !important;
    background-color: var(--accent) !important;
    color: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 13px 20px !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
    transition: background 0.15s ease !important;
}

.primary-btn:hover {
    background-color: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
}

.primary-btn:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}

.primary-btn[disabled] {
    opacity: 0.6 !important;
    cursor: not-allowed !important;
}

/* Status message */
.status-container {
    font-size: 13px;
    margin-top: 12px;
    min-height: 20px;
    text-align: center;
}

.status-warning {
    color: var(--text-danger);
}

.status-info {
    color: var(--text-secondary);
}

/* Readout Box */
.readout-box {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border-subtle);
}

.readout-item {
    background-color: var(--bg-input);
    border: 1px solid var(--border-input);
    border-radius: 8px;
    padding: 10px 14px;
}

.readout-label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.readout-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
}

/* Footer */
.aura-footer {
    text-align: center;
    padding: 32px 16px 0;
    color: var(--text-muted);
    font-size: 13px;
    border-top: 1px solid var(--border-subtle);
    margin-top: 32px;
}

/* Responsive */
@media (max-width: 768px) {
    .main-row {
        flex-direction: column !important;
    }
    .right-column {
        position: static !important;
    }
    .readout-box {
        grid-template-columns: 1fr;
    }
}
"""

def process_video_pipeline(
    video_input,
    style_choice: str,
    progress=gr.Progress(track_tqdm=False)
):
    """
    Executes the video processing pipeline.
    Maintains clean error handling and plain-text status outputs without emoji.
    """
    if not video_input:
        return (
            None,
            '<div class="readout-value">None</div>',
            '<div class="readout-value">None</div>',
            '<div class="status-container"><span class="status-warning">Add a clip or choose a sample first.</span></div>'
        )

    session_temp_dir = tempfile.mkdtemp(prefix="aura_session_")
    t_start = time.time()

    try:
        progress(0.1, desc="Processing... this takes about 20 seconds")
        logger.info(f"Processing video: input='{video_input}', style='{style_choice}'")

        # 1. Extract Frames
        frames, meta = extract_clip_frames(video_input, target_num_frames=12, max_dimension=720)

        # 2. Detect Face and Zoom Target
        progress(0.3, desc="Processing... this takes about 20 seconds")
        zoom_target, boxes = detect_face_and_zoom_target(frames)
        face_crop = get_best_face_crop(frames, boxes)

        # 3. Emotion Analysis
        progress(0.5, desc="Processing... this takes about 20 seconds")
        raw_emotion, confidence, mapped_style = analyze_emotion(face_crop)

        # Map style
        if style_choice == "Auto":
            chosen_style = mapped_style
        else:
            chosen_style = style_choice.upper().strip()

        # 4. Compose Video
        progress(0.7, desc="Processing... this takes about 20 seconds")
        output_mp4 = os.path.join(session_temp_dir, f"aura_{chosen_style.lower()}_{int(time.time())}.mp4")

        compose_aura_video(
            frames=frames,
            zoom_target=zoom_target,
            style_name=chosen_style,
            output_path=output_mp4,
            output_width=720,
            output_height=1280,
            fps=24,
            duration=6.0
        )

        progress(1.0, desc="Processing complete.")
        total_time = time.time() - t_start
        logger.info(f"Generated aura edit in {total_time:.2f}s: {output_mp4}")

        detected_expression_text = f'<div class="readout-value">{raw_emotion.capitalize()} ({int(confidence * 100)}%)</div>'
        selected_style_text = f'<div class="readout-value">{chosen_style.capitalize()}</div>'
        status_msg = f'<div class="status-container"><span class="status-info">Generated in {total_time:.1f}s</span></div>'

        return output_mp4, detected_expression_text, selected_style_text, status_msg

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        logger.error(traceback.format_exc())
        return (
            None,
            '<div class="readout-value">None</div>',
            '<div class="readout-value">None</div>',
            '<div class="status-container"><span class="status-warning">An error occurred while processing the video. Please try again.</span></div>'
        )

    finally:
        # Ephemeral privacy cleanup
        try:
            if video_input and "aura_smoke_test" in video_input and os.path.exists(video_input):
                os.remove(video_input)
        except Exception:
            pass

def load_demo_sample():
    """Loads bundled sample portrait clip."""
    if os.path.exists(SAMPLE_CLIP_PATH):
        return SAMPLE_CLIP_PATH, "Auto", '<div class="status-container"></div>'
    return None, "Auto", '<div class="status-container"></div>'

def clear_inputs():
    """Clears inputs and resets outputs to default empty state."""
    return None, '<div class="readout-value">None</div>', '<div class="readout-value">None</div>', '<div class="status-container"></div>'

def create_ui():
    """Builds and returns the minimal, professional Gradio Blocks interface."""
    theme = gr.themes.Base(
        primary_hue="rose",
        secondary_hue="zinc",
        neutral_hue="zinc",
    )

    with gr.Blocks(title="AURA EDIT") as demo:
        with gr.Column(elem_classes=["aura-container"]):
            # Header
            gr.HTML("""
            <div class="aura-header">
                <h1 class="aura-title">AURA EDIT</h1>
                <p class="aura-subtitle">Your 6-second aura, generated by AI</p>
            </div>
            """)

            with gr.Row(elem_classes=["main-row"]):
                # Left Column: Input and Style Controls
                with gr.Column(scale=5):
                    # Card 1: Input
                    with gr.Column(elem_classes=["aura-card"]):
                        gr.HTML('<div class="card-title">Input</div>')
                        video_input = gr.Video(
                            sources=["webcam", "upload"],
                            show_label=False,
                            interactive=True,
                            elem_classes=["compact-video"]
                        )
                        gr.HTML('<div class="card-helper">Record a 3-4 second clip or upload a video file.</div>')

                        with gr.Row(elem_id="input-actions-row", equal_height=True):
                            use_sample_btn = gr.Button("Use sample clip", elem_classes=["secondary-btn"])
                            clear_btn = gr.Button("Clear", elem_classes=["secondary-btn"])

                    # Card 2: Style
                    with gr.Column(elem_classes=["aura-card"]):
                        gr.HTML('<div class="card-title">Style</div>')
                        style_dropdown = gr.Dropdown(
                            choices=["Auto", "Dark", "Gold", "Fire"],
                            value="Auto",
                            show_label=False,
                            container=False,
                            interactive=True
                        )
                        gr.HTML('<div class="card-helper">Auto selects the visual style based on your facial expression.</div>')

                    # Primary Button & Inline Status
                    generate_btn = gr.Button(
                        "Generate",
                        elem_classes=["primary-btn"],
                        interactive=True
                    )

                    status_display = gr.HTML(
                        '<div class="status-container"></div>',
                        elem_classes=["status-container"]
                    )

                # Right Column: Result (Sticky)
                with gr.Column(scale=5, elem_classes=["right-column"]):
                    with gr.Column(elem_classes=["aura-card"]):
                        gr.HTML('<div class="card-title">Result</div>')
                        output_video = gr.Video(
                            label="Aura video",
                            show_label=True,
                            autoplay=True,
                            interactive=False
                        )

                        # Small two-field readout underneath
                        with gr.Row(elem_classes=["readout-box"]):
                            with gr.Column(elem_classes=["readout-item"]):
                                gr.HTML('<div class="readout-label">Detected expression</div>')
                                expression_readout = gr.HTML('<div class="readout-value">None</div>')
                            with gr.Column(elem_classes=["readout-item"]):
                                gr.HTML('<div class="readout-label">Selected style</div>')
                                style_readout = gr.HTML('<div class="readout-value">None</div>')

            # Footer
            gr.HTML("""
            <div class="aura-footer">
                No data is stored. Your video is deleted after processing.
            </div>
            """)

        # Event Handlers
        use_sample_btn.click(
            fn=load_demo_sample,
            inputs=[],
            outputs=[video_input, style_dropdown, status_display]
        )

        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[video_input, expression_readout, style_readout, status_display]
        )

        generate_btn.click(
            fn=process_video_pipeline,
            inputs=[video_input, style_dropdown],
            outputs=[output_video, expression_readout, style_readout, status_display]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    theme = gr.themes.Base(
        primary_hue="rose",
        secondary_hue="zinc",
        neutral_hue="zinc",
    )
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            theme=theme,
            css=CUSTOM_CSS,
            share=False,
            show_error=False
        )
    except OSError:
        demo.launch(
            server_name="127.0.0.1",
            server_port=None,
            theme=theme,
            css=CUSTOM_CSS,
            share=False,
            show_error=False
        )
