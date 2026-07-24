"""Gradio web UI for the multi-agent supervisor.

Users can type a message or record their voice. The app shows the supervisor's
reasoning trace (the `[Supervisor]` lines) and renders the final answer in blue.

Run:  python3 app.py
"""

from html import escape
import gradio as gr
from VoiceAssistant import build_default_assistant

# Build the wired assistant once and reuse it across requests.
assistant = build_default_assistant()
supervisor = assistant.supervisor


def _trace_html(lines):
    if not lines:
        return ""
    body = "<br>".join(escape(line) for line in lines)
    return (
        "<div style='font-family:monospace;font-size:0.9em;color:#6b7280;"
        "background:#f5f5f7;border-radius:8px;padding:12px;white-space:pre-wrap'>"
        f"{body}</div>"
    )


def _answer_html(answer):
    body = escape(answer).replace("\n", "<br>")
    return (
        "<div style='color:#1e6fff;font-size:1.1em;line-height:1.5;"
        "font-weight:500'>"
        f"{body}</div>"
    )


def respond(text, audio):
    """Resolve the query (typed text wins, else transcribe the recording), run
    the supervisor, and return (resolved query, trace HTML, blue answer HTML,
    spoken-answer .wav path)."""
    query = (text or "").strip()
    if not query and audio:
        query = assistant.transcribe_file(audio)

    if not query:
        return (
            "",
            "",
            _answer_html("Please type a message or record your voice."),
            None,
        )

    trace = []
    answer = supervisor.route(query, on_event=trace.append)
    voice_path = assistant.synthesize_to_file(answer)
    return query, _trace_html(trace), _answer_html(answer), voice_path


with gr.Blocks(title="Multi-Agent Supervisor!!") as demo:
    gr.Markdown(
        "# Multi-Agent Searcher !!\n"
        "Type a message or record your voice. The supervisor plans, picks the "
        "right agent(s), and answers."
    )

    with gr.Row():
        text_in = gr.Textbox(
            label="Type your message",
            placeholder="e.g. What is the latest news about SpaceX?",
            scale=3,
        )
        audio_in = gr.Audio(
            sources=["microphone"], type="filepath", label="…or speak", scale=2
        )

    submit = gr.Button("Ask", variant="primary")

    query_out = gr.Textbox(label="You asked", interactive=False)
    trace_out = gr.HTML(label="Supervisor trace")
    answer_out = gr.HTML(label="Answer")
    answer_audio = gr.Audio(label="Answer (voice)", autoplay=True)

    outputs = [query_out, trace_out, answer_out, answer_audio]
    submit.click(respond, inputs=[text_in, audio_in], outputs=outputs)
    text_in.submit(respond, inputs=[text_in, audio_in], outputs=outputs)


if __name__ == "__main__":
    demo.launch()
