# GitHub 协作操作指南

## 第一步：创建GitHub仓库

### 1.1 在GitHub网站上操作

1. 登录 https://github.com
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息：
   - Repository name: `grid-trading-backtest` (或自定义名称)
   - Description: `A股网格交易策略回测系统`
   - 选择 **Public** 或 **Private**（推荐Private用于课程项目）
   - **不要**勾选 "Add a README file"（我们已经有了）
   - **不要**勾选 "Add .gitignore"（我们已经有了）
4. 点击 "Create repository"

### 1.2 记录仓库地址
创建后会显示类似这样的地址：
```
https://github.com/你的用户名/grid-trading-backtest.git
```

## 第二步：本地初始化并推送

打开PowerShell或命令行，**进入my-grid目录**执行：

```bash
# 进入my-grid目录（这将成为仓库根目录）
cd D:\研究生课\cs277\my-grid

# 初始化git仓库（如果还没有初始化）
git init

# 将README等文件复制到my-grid目录
# 从上层目录复制：README.md, requirements.txt, .gitignore, COLLABORATION_GUIDE.md

# 添加所有文件到暂存区
git add .

# 查看将要提交的文件（可选，用于检查）
git status

# 提交到本地仓库
git commit -m "Initial commit: 网格交易回测系统"

# 添加远程仓库地址（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/grid-trading-backtest.git

# 推送到GitHub（首次推送）
git push -u origin main
```

如果提示分支名是master而不是main：
```bash
# 重命名分支为main
git branch -M main
# 再推送
git push -u origin main
```

## 第三步：邀请协作者

### 3.1 添加协作者

1. 在GitHub仓库页面，点击 "Settings" 标签
2. 左侧菜单选择 "Collaborators"
3. 点击 "Add people"
4. 输入协作者的GitHub用户名或邮箱
5. 选择权限级别（推荐 **Write** 权限）
6. 发送邀请

### 3.2 协作者接受邀请

协作者会收到邮件通知，或在GitHub通知中看到邀请，点击接受即可。

## 第四步：协作者克隆仓库

协作者在自己的电脑上执行：

```bash
# 克隆仓库
git clone https://github.com/你的用户名/grid-trading-backtest.git

# 进入项目目录
cd grid-trading-backtest

# 安装依赖
pip install -r requirements.txt

# 准备数据文件（需要单独提供数据压缩包）
# 解压到 上证信息数据2024/his_sh1_201907-202406/
```

## 第五步：日常协作工作流

### 5.1 每次开始工作前（拉取最新代码）

```bash
# 拉取最新代码
git pull origin main
```

### 5.2 进行修改

修改代码、运行实验、生成结果...

### 5.3 提交更改

```bash
# 查看修改的文件
git status

# 添加修改的文件到暂存区
git add .
# 或只添加特定文件
git add my-grid/example_atr_adaptive_grid.py

# 提交到本地仓库（写清楚修改内容）
git commit -m "实验：测试000008股票在不同ATR参数下的表现"

# 推送到GitHub
git push origin main
```

### 5.4 提交消息规范（建议）

```bash
git commit -m "实验：xxx股票回测"
git commit -m "修复：中文显示乱码问题"
git commit -m "优化：提高数据加载速度"
git commit -m "文档：更新README说明"
```

## 第六步：分支协作（高级，可选）

如果实验较多，建议使用分支避免冲突：

### 6.1 创建实验分支

```bash
# 创建并切换到新分支
git checkout -b experiment/atr-optimization

# 在这个分支上工作...
# 修改代码、运行实验

# 提交更改
git add .
git commit -m "实验：ATR参数优化"

# 推送分支到GitHub
git push origin experiment/atr-optimization
```

### 6.2 合并到主分支

实验完成后，在GitHub上创建Pull Request：

1. 在仓库页面点击 "Pull requests" 标签
2. 点击 "New pull request"
3. 选择 base: main ← compare: experiment/atr-optimization
4. 填写PR描述，说明实验内容和结果
5. 点击 "Create pull request"
6. 审查后点击 "Merge pull request"

或者在本地合并：

```bash
# 切换回主分支
git checkout main

# 合并实验分支
git merge experiment/atr-optimization

# 推送到GitHub
git push origin main
```

## 第七步：查看协作者的工作

### 7.1 查看提交历史

```bash
# 查看提交历史
git log --oneline --graph --all

# 查看某次提交的详细修改
git show <commit-hash>
```

### 7.2 在GitHub网站查看

1. 在仓库页面点击 "Commits" 查看所有提交
2. 点击具体提交查看修改内容
3. 可以添加评论进行讨论

## 常见问题

### Q1: 推送时要求输入用户名密码

A: GitHub已不支持密码认证，需要使用Personal Access Token：

1. GitHub右上角头像 → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → 勾选 `repo` 权限 → Generate token
3. 复制token（只显示一次！）
4. 推送时用token替代密码

或者配置SSH密钥（推荐）：
```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 复制公钥内容
cat ~/.ssh/id_ed25519.pub

# 在GitHub Settings → SSH and GPG keys → New SSH key 中添加

# 更改远程仓库地址为SSH
git remote set-url origin git@github.com:你的用户名/grid-trading-backtest.git
```

### Q2: 出现合并冲突怎么办？

A: 如果两人同时修改了同一文件：

```bash
# 拉取时提示冲突
git pull origin main

# Git会标记冲突文件，打开文件会看到：
<<<<<<< HEAD
你的修改
=======
别人的修改
>>>>>>> branch-name

# 手动解决冲突，保留需要的内容，删除标记
# 然后：
git add 冲突文件.py
git commit -m "解决合并冲突"
git push origin main
```

### Q3: 如何共享数据文件？

A: 数据文件太大不适合上传GitHub，建议：

1. **网盘分享**：百度云、OneDrive等
2. **Git LFS**：适合较大文件
   ```bash
   git lfs install
   git lfs track "*.csv"
   git add .gitattributes
   ```
3. **分享链接**：在README中添加数据下载链接

### Q4: 如何撤销提交？

A: 如果提交了错误的内容：

```bash
# 撤销最后一次提交，保留修改
git reset --soft HEAD~1

# 撤销最后一次提交，丢弃修改（危险！）
git reset --hard HEAD~1

# 如果已经推送到GitHub，需要强制推送（慎用！）
git push -f origin main
```

## 实验记录模板

建议在每次实验后更新一个 `EXPERIMENTS.md` 文件：

```markdown
# 实验记录

## 实验1：ATR策略vs固定网格策略对比
- 日期：2024-06-20
- 操作者：张三
- 股票代码：000008
- 参数设置：
  - 选股期：2020-01-01 ~ 2022-12-31
  - 回测期：2023-01-01 ~ 2024-06-20
  - ATR乘数：1.0
- 结果：
  - 年化收益率：10.35%
  - 夏普比率：0.31
  - 最大回撤：-12.45%
- 结论：ATR策略在该股票上优于固定网格

## 实验2：...
```

## 快速命令参考

```bash
# 查看状态
git status

# 拉取最新代码
git pull

# 添加所有修改
git add .

# 提交
git commit -m "说明"

# 推送
git push

# 查看历史
git log --oneline

# 创建分支
git checkout -b branch-name

# 切换分支
git checkout main

# 查看远程仓库
git remote -v
```

## 推荐工具

- **GitHub Desktop**：可视化Git客户端，适合不熟悉命令行的用户
- **VS Code**：内置Git支持，有可视化界面
- **GitKraken**：强大的Git可视化工具

## 需要帮助？

- [Git官方文档](https://git-scm.com/doc)
- [GitHub官方指南](https://docs.github.com/cn)
- 在仓库的Issues中提问
