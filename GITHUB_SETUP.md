# GitHub快速设置指南 - 只上传my-grid目录

## 📦 已完成的准备工作

✅ README.md - 已复制到my-grid目录
✅ requirements.txt - 已复制到my-grid目录  
✅ .gitignore - 已复制到my-grid目录
✅ COLLABORATION_GUIDE.md - 已复制到my-grid目录

## 🚀 推送到GitHub的步骤

### 方法1：在PowerShell中执行（推荐）

```powershell
# 1. 进入my-grid目录
cd D:\研究生课\cs277\my-grid

# 2. 初始化git
git init

# 3. 添加所有文件
git add .

# 4. 提交
git commit -m "Initial commit: 网格交易回测系统"

# 5. 在GitHub上创建仓库后，添加远程地址（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/grid-trading-backtest.git

# 6. 重命名分支为main
git branch -M main

# 7. 推送
git push -u origin main
```

### 方法2：使用GitHub Desktop（更简单）

1. 打开GitHub Desktop
2. File → Add Local Repository
3. 选择 `D:\研究生课\cs277\my-grid` 目录
4. 点击"Create repository"
5. Commit所有文件
6. Publish repository到GitHub

## 📁 最终GitHub仓库结构

```
grid-trading-backtest/              # 仓库根目录（对应本地my-grid目录）
├── README.md                       # 项目说明
├── requirements.txt                # Python依赖
├── .gitignore                      # 忽略文件配置
├── COLLABORATION_GUIDE.md          # 协作指南
├── data_clean.py                   # 数据清洗
├── grid.py                         # 网格策略核心
├── select_stock.py                 # 标的筛选
├── data_vis.py                     # 数据可视化
├── example_atr_adaptive_grid.py    # ATR策略示例
├── example_correct_workflow.py     # 完整工作流
└── grid2.py                        # 简化实现

注意：以下不会上传到GitHub（在.gitignore中）
├── 上证信息数据2024/              # 数据文件（太大）
├── *.csv                           # 交易记录
├── *.png                           # 图表
└── __pycache__/                    # Python缓存
```

## 👥 邀请协作者

1. 在GitHub仓库页面 → Settings → Collaborators
2. Add people → 输入GitHub用户名
3. 选择权限：Write

## 📤 数据文件分享方案

由于数据文件太大（不在GitHub仓库中），需要单独分享：

### 选项1：网盘分享
```
1. 压缩 my-grid/上证信息数据2024/ 目录
2. 上传到百度云/OneDrive/Google Drive
3. 在README中添加下载链接
4. 协作者下载后解压到项目根目录
```

### 选项2：在README中添加说明
在README.md的"数据准备"部分添加：
```markdown
### 3. 数据准备

数据文件请通过以下方式获取：
- 下载链接：[百度云链接](https://pan.baidu.com/xxx)
- 提取码：xxxx

下载后解压到项目根目录，最终结构：
├── 上证信息数据2024/
│   └── his_sh1_201907-202406/
│       ├── 20191113/
│       │   └── Day.csv
│       └── ...
```

## ✅ 检查清单

推送前检查：
- [ ] 已进入my-grid目录
- [ ] README.md等配置文件在my-grid目录中
- [ ] 已在GitHub创建空仓库
- [ ] 数据文件已被.gitignore排除（不会上传）

推送后检查：
- [ ] GitHub仓库中能看到所有.py文件
- [ ] README.md显示正常
- [ ] 没有数据文件被上传
- [ ] 协作者已收到邀请

## 🔧 协作者设置步骤

协作者收到邀请后：

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/grid-trading-backtest.git

# 2. 进入目录
cd grid-trading-backtest

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载数据文件（通过你提供的链接）
# 解压到当前目录，确保有 上证信息数据2024/ 文件夹

# 5. 测试运行
python example_atr_adaptive_grid.py

# 6. 开始实验！
```

## 💡 提示

- 如果需要修改配置文件，在 `D:\研究生课\cs277\my-grid\` 目录中修改
- 协作者运行实验生成的CSV和PNG不会被提交（已在.gitignore中）
- 建议在提交消息中描述实验结果
- 可以使用GitHub Issues管理实验任务

需要帮助？查看 COLLABORATION_GUIDE.md 获取详细说明！
