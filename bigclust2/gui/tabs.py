from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QCursor, QMouseEvent
from PySide6.QtWidgets import QApplication, QTabBar, QTabWidget


class ViewTabBar(QTabBar):
    """Tab bar that requests a detach when a tab is dragged out vertically.

    Horizontal dragging keeps QTabBar's built-in reordering; only a clear
    vertical exit from the bar is interpreted as "tear this tab off".
    """

    detach_requested = Signal(int, QPoint)  # tab index, global cursor position

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pressed_index = -1

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._pressed_index = self.tabAt(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pressed_index >= 0 and event.buttons() & Qt.LeftButton:
            pos = event.position().toPoint()
            threshold = max(
                int(self.height() * 1.5), QApplication.startDragDistance() * 4
            )
            if pos.y() < -threshold or pos.y() > self.height() + threshold:
                # Pressing a tab makes it current, and reordering during the
                # drag moves it with the cursor - so currentIndex() is the
                # dragged tab even if the press index went stale.
                index = self.currentIndex()
                self._pressed_index = -1
                # End QTabBar's internal move before the tab is pulled out from
                # under it.
                self._finish_native_drag(pos)
                self.detach_requested.emit(index, QCursor.pos())
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._pressed_index = -1
        super().mouseReleaseEvent(event)

    def _finish_native_drag(self, pos):
        release = QMouseEvent(
            QEvent.MouseButtonRelease,
            QPointF(pos),
            QPointF(self.mapToGlobal(pos)),
            Qt.LeftButton,
            Qt.NoButton,
            Qt.NoModifier,
        )
        super().mouseReleaseEvent(release)


class ViewTabWidget(QTabWidget):
    """Tab widget holding one view (MainWidget) per tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabBar(ViewTabBar(self))
        self.setDocumentMode(True)
        self.setTabsClosable(True)
        self.setMovable(True)
        # With a single tab the bar is hidden and the window looks like before.
        self.setTabBarAutoHide(True)

    @property
    def detach_requested(self):
        return self.tabBar().detach_requested
