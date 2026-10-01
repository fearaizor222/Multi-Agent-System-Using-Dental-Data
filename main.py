"""Entry point for the Dental Multi-Agent System."""

import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from langchain_core.messages import HumanMessage
from src.graph.workflow import build_dental_agent_graph
from src.core.config import settings

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    console = Console(legacy_windows=False)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


def print_banner():
    """Display startup banner."""
    title = f"🦷 {settings.PROJECT_NAME} (v{settings.VERSION}) 🦷"
    subtitle = (
        f"Framework: LangChain & LangGraph | "
        f"LLM Provider: {settings.LLM_PROVIDER.upper()} | "
        f"Retrieval: Hybrid RAG (Dense + BM25 with RRF)"
    )
    if HAS_RICH:
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]", border_style="cyan"))
    else:
        print("=" * 60)
        print(title)
        print(subtitle)
        print("=" * 60)


def run_query(app, query: str):
    """Run a single query through the LangGraph Multi-Agent workflow."""
    if HAS_RICH:
        console.print(f"\n[bold yellow]👤 Câu hỏi người dùng:[/bold yellow] [white]{query}[/white]")
        console.print("[dim]Đang kích hoạt quy trình điều phối Multi-Agent...[/dim]\n")
    else:
        print(f"\n👤 Câu hỏi người dùng: {query}\n")

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next": "",
        "rag_context": [],
        "patient_data": {},
        "step_count": 0,
    }

    consultant_content = "Không tìm thấy nội dung phản hồi."
    step_num = 1

    # Stream execution step-by-step
    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            if HAS_RICH:
                color = "magenta" if node_name == "supervisor" else ("blue" if node_name == "rag_retriever" else "green")
                console.print(f"[bold {color}]▶ Bước {step_num}: Node [{node_name}][/bold {color}]")
            else:
                print(f"▶ Bước {step_num}: Node [{node_name}]")

            if "messages" in state_update and state_update["messages"]:
                last_msg = state_update["messages"][-1]
                sender = getattr(last_msg, "name", node_name)
                content_preview = last_msg.content[:140] + ("..." if len(last_msg.content) > 140 else "")
                if HAS_RICH:
                    console.print(f"  [dim]Tin nhắn từ {sender}:[/dim] {content_preview}")
                else:
                    print(f"  Tin nhắn từ {sender}: {content_preview}")

                if sender == "dental_consultant" or node_name == "dental_consultant":
                    consultant_content = last_msg.content

            step_num += 1

    # Print final consultation response
    if HAS_RICH:
        console.print("\n" + "=" * 60)
        console.print(Panel("[bold green]🩺 KẾT QUẢ TƯ VẤN TỪ DENTAL CONSULTANT AGENT[/bold green]", border_style="green"))
        console.print(Markdown(consultant_content))
    else:
        print("\n" + "=" * 60)
        print("🩺 KẾT QUẢ TƯ VẤN:")
        print(consultant_content)


def main():
    """Main CLI entrypoint."""
    print_banner()

    # Compile the LangGraph agent graph
    app = build_dental_agent_graph()

    sample_query = (
        "Răng số 6 của tôi bị đau buốt dữ dội khi uống nước đá lạnh, "
        "ban đêm còn nhức nhối thành từng cơn. Bác sĩ cho hỏi tôi có thể bị bệnh gì và cần làm gì?"
    )

    if len(sys.argv) > 1:
        # Query passed as command line argument
        user_query = " ".join(sys.argv[1:])
        run_query(app, user_query)
    else:
        # Run demo sample query first
        if HAS_RICH:
            console.print("[cyan]Chạy câu hỏi mẫu thử nghiệm tự động:[/cyan]")
        run_query(app, sample_query)

        # Enter interactive loop
        print("\n" + "-" * 60)
        print("Nhập câu hỏi của bạn (hoặc gõ 'exit' / 'quit' để thoát):")
        while True:
            try:
                user_input = input("\n[Bạn] > ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Tạm biệt! Chúc bạn một ngày tốt lành.")
                    break
                if not user_input:
                    continue
                run_query(app, user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nĐã dừng chương trình.")
                break


if __name__ == "__main__":
    main()
