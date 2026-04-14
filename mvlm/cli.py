from __future__ import annotations

import sys


def main():
    args = sys.argv[1:]
    if not args or args[0] == "show":
        from mvlm.web import show

        port = 8765
        if len(args) > 1:
            try:
                port = int(args[1])
            except ValueError:
                pass
        show(port=port)
    elif args[0] == "report":
        from mvlm.results import report

        project = args[1] if len(args) > 1 else None
        report(project=project)
    else:
        print("Usage: smollest [show|report] [project]")
        sys.exit(1)


if __name__ == "__main__":
    main()
