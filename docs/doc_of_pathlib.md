# help doc of pathlib
```python
q = Path("results", "exp1.log")      # 相对路径（自动拼接）

p.name        # 文件名 exp1.log
p.stem        # 无后缀 exp1
p.suffix      # 后缀 .log
p.parent      # 上级目录 /home/youwei/data
p.parts       # 拆成元组 ('/', 'home', 'youwei', 'data')
p.exists()    # 路径是否存在 → True/False
p.is_file()   # 是否是文件 → True/False
p.is_dir()    # 是否是目录 → True/False


# read/write file
(Path("a.txt")).write_text("hello pathlib")
text = (Path("a.txt")).read_text()

##拼接
cfg = root / "config" / "train.yaml"   # 用 / 就能拼，跨平台

##file_operate
new_dir = Path("tmp")
new_dir.mkdir(parents=True, exist_ok=True)  # 递归创建
(new_dir / "demo.txt").touch()              # 创建空文件
(new_dir / "demo.txt").unlink()             # 删除文件
new_dir.rmdir()                             # 删除空目录
```