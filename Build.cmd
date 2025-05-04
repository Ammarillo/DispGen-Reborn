.\DispGen\Scripts\activate

pyinstaller --onefile --windowed --name "DispGen-Reborn" --hidden-import=pyqtgraph.opengl --hidden-import=OpenGL --hidden-import=OpenGL.GL --hidden-import=OpenGL.GLU --icon="resources\D-Gen_Icon.ico" main.py

pyinstaller --clean DispGen-Reborn.spec