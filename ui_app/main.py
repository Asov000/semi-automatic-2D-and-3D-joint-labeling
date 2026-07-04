# -*- coding: utf-8 -*-
"""PyQt 应用入口模块，创建并运行标注主窗口。"""

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from ui_app.annotator import SAMAnnotator
    from ui_app.context import QApplication
else:
    from .annotator import SAMAnnotator
    from .context import QApplication, sys


def run_app() -> int:
    """创建 Qt 应用实例，显示主窗口并进入事件循环。"""
    app = QApplication(sys.argv)
    window = SAMAnnotator()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(run_app())
