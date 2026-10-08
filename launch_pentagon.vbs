Set WshShell = CreateObject("WScript.Shell")
' 0 = ウィンドウを完全に非表示にして実行
' False = プログラムの終了を待たずに次の処理へ
WshShell.Run "python run_pentagon_gui.py", 0, False
