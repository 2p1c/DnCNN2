import subprocess
import sys
import os

def run(cmd):
    try:
        # 兼容 Windows/Linux/Mac
        subprocess.run(cmd, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing: {cmd}")
        sys.exit(1)

def main():
    if not os.path.exists(".git"):
        print("⚠️ Not a git repository.")
        return

    # 1. 检查状态
    status = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True).stdout.strip()
    if not status:
        print("✅ No changes to commit.")
        return

    # 2. 获取或生成消息
    msg = sys.argv[1] if len(sys.argv) > 1 else "wip: auto save progress"
    
    # 3. 执行
    print(f"📦 Staging all files...")
    run("git add .")
    
    print(f"📝 Committing: {msg}")
    run(f'git commit -m "{msg}"')
    
    print(f"🚀 Pushing...")
    try:
        run("git push")
        print("✅ Done!")
    except:
        print("⚠️ Push failed. Check your remote/branch upstream.")

if __name__ == "__main__":
    main()
