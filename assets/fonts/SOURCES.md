# 状态栏像素字形

`uibattlesmall.png` / `uibattlesmall.xml` 来自 Create Your Frisk 的公开默认资源：

- 上游仓库：https://github.com/RhenaudTheLukark/CreateYourFrisk
- 资源目录：`Assets/Default/Sprites/UI/Fonts/`
- 仓库许可：https://github.com/RhenaudTheLukark/CreateYourFrisk/blob/master/LICENSE
- 字体作者 Anna Anthropy 的字体包：https://w.itch.io/world-of-fonts

图集和坐标表保留上游原文件。渲染器按图集逐字贴图，不调用系统字体；字形坐标采用 Unity 的左下角原点，并转换为 Pillow 的左上角原点。

四个底部按钮直接使用用户提供的 `sprite/*1.png`，不重新排字。
