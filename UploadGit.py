import subprocess
import os
from datetime import datetime
from pathlib import Path
import sys

# Windows 콘솔(cp949)에서도 유니코드 출력 때문에 죽지 않게 방어
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

class GitUploader:
    def __init__(self, repo_path, remote_url_with_token, branch='main'):
        """
        Git 업로더 초기화 (토큰 포함)
        
        Args:
            repo_path: Git 저장소 경로
            remote_url_with_token: 토큰이 포함된 Git URL
            branch: 브랜치 이름
        """
        self.repo_path = os.path.abspath(repo_path)
        self.remote_url = remote_url_with_token
        self.branch = branch
        
        # 디렉토리가 없으면 생성
        Path(self.repo_path).mkdir(parents=True, exist_ok=True)
        
    def run_command(self, command, check_error=True, show_output=True):
        """Git 명령어 실행"""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            output = result.stdout + result.stderr
            
            if show_output and output.strip():
                print(output.strip())
            
            if check_error and result.returncode != 0:
                # 일부 경고는 무시
                if "nothing to commit" not in output and "up to date" not in output.lower():
                    if "already exists" not in output:  # 이미 존재하는 경우는 무시
                        print(f"경고: {output}")
                        return False, output
            
            return True, output
            
        except Exception as e:
            print(f"예외 발생: {str(e)}")
            return False, str(e)
    
    def check_git_installed(self):
        """Git 설치 확인"""
        success, _ = self.run_command("git --version", check_error=False, show_output=False)
        if not success:
            print("Git이 설치되어 있지 않습니다. https://git-scm.com/ 에서 설치해주세요.")
            return False
        return True
    
    def is_git_repo(self):
        """Git 저장소 여부 확인"""
        git_dir = os.path.join(self.repo_path, '.git')
        return os.path.exists(git_dir)
    
    def check_remote_exists(self):
        """원격 저장소 존재 확인"""
        success, output = self.run_command("git remote -v", check_error=False, show_output=False)
        return success and "origin" in output
    
    def setup_remote(self):
        """원격 저장소 설정"""
        print("\n🔗 원격 저장소 설정 중...")
        
        # 기존 origin 제거 (있다면)
        self.run_command("git remote remove origin", check_error=False, show_output=False)
        
        # 새로운 origin 추가
        success, output = self.run_command(
            f'git remote add origin "{self.remote_url}"',
            check_error=False
        )
        
        if success or "already exists" in output:
            print(" 원격 저장소 설정 완료")
            
            # 원격 저장소 확인
            self.run_command("git remote -v", show_output=True)
            return True
        
        print(" 원격 저장소 설정 실패")
        return False
    
    def get_current_branch(self):
        """현재 브랜치 확인"""
        success, output = self.run_command("git branch --show-current", check_error=False, show_output=False)
        if success and output.strip():
            return output.strip()
        
        # 대체 방법
        success, output = self.run_command("git rev-parse --abbrev-ref HEAD", check_error=False, show_output=False)
        if success and output.strip():
            return output.strip()
        
        return None
    
    def clone_repo(self):
        """저장소 클론"""
        print(f"\n 저장소 클론: {self.repo_path}")
        
        # 부모 디렉토리로 이동
        parent_dir = os.path.dirname(self.repo_path)
        folder_name = os.path.basename(self.repo_path)
        
        command = f'git clone "{self.remote_url}" "{folder_name}"'
        
        result = subprocess.run(
            command,
            cwd=parent_dir,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"클론 실패: {result.stderr}")
            return False
        
        print("클론 완료")
        return True
    
    def init_or_clone(self):
        """저장소 초기화 또는 클론"""
        if self.is_git_repo():
            print("기존 Git 저장소 사용")
            
            # 원격 저장소 확인 및 설정
            if not self.check_remote_exists():
                print("원격 저장소가 설정되지 않았습니다.")
                if not self.setup_remote():
                    return False
            else:
                print("원격 저장소 이미 설정됨")
            
            # 현재 브랜치 확인
            current_branch = self.get_current_branch()
            if current_branch:
                print(f"현재 브랜치: {current_branch}")
                self.branch = current_branch
            
            return True
        
        # 폴더가 비어있으면 클론
        if not os.listdir(self.repo_path):
            if self.clone_repo():
                # 클론 후 브랜치 확인
                current_branch = self.get_current_branch()
                if current_branch:
                    self.branch = current_branch
                    print(f"브랜치 설정: {self.branch}")
                return True
            return False
        
        # 폴더에 파일이 있으면 초기화
        print("Git 저장소 초기화...")
        self.run_command("git init")
        
        # 원격 저장소 설정
        if not self.setup_remote():
            return False
        
        # 초기 커밋 생성 (파일이 있는 경우)
        self.run_command("git add .", check_error=False)
        self.run_command('git commit -m "Initial commit"', check_error=False)
        
        # 브랜치 확인 및 설정
        current_branch = self.get_current_branch()
        if current_branch:
            self.branch = current_branch
        else:
            # 기본 브랜치 생성
            self.run_command(f"git checkout -b {self.branch}", check_error=False)
        
        print(f"브랜치 설정: {self.branch}")
        
        return True
    
    def pull(self):
        """원격 저장소에서 변경사항 가져오기"""
        print("\nPull 실행...")
        
        # fetch
        self.run_command("git fetch origin", check_error=False)
        
        # 원격 브랜치 존재 확인
        success, output = self.run_command(
            f"git ls-remote --heads origin {self.branch}",
            check_error=False,
            show_output=False
        )
        
        if not output.strip():
            print(f"원격에 '{self.branch}' 브랜치가 없습니다. 첫 push를 진행합니다.")
            return True
        
        # pull with rebase
        success, output = self.run_command(
            f"git pull origin {self.branch} --rebase",
            check_error=False
        )
        
        if "up to date" in output.lower():
            print("이미 최신 상태입니다.")
        elif success:
            print("Pull 성공")
        
        return True
    
    def add_all(self):
        """모든 파일 추가"""
        print("\n파일 추가...")
        return self.run_command("git add .")
    
    def commit(self, message=None):
        """커밋"""
        if message is None:
            message = f"Auto update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        print(f"\n커밋: {message}")
        success, output = self.run_command(f'git commit -m "{message}"', check_error=False)
        
        if "nothing to commit" in output:
            print("변경사항이 없습니다.")
            return True
        
        return success
    
    def push(self):
        """푸시"""
        print(f"\nPush 실행 (브랜치: {self.branch})...")
        
        # 원격 저장소 재확인
        if not self.check_remote_exists():
            print("원격 저장소가 없습니다. 재설정 중...")
            if not self.setup_remote():
                return False
        
        # upstream 설정과 함께 push
        success, output = self.run_command(
            f"git push -u origin {self.branch}",
            check_error=False
        )
        
        if success or "up-to-date" in output.lower():
            print("Push 성공")
            return True
        
        # 실패 시 상세 정보 출력
        print(f"Push 실패:")
        print(output)
        
        return False
    
    def status(self):
        """상태 확인"""
        print("\nGit 상태:")
        return self.run_command("git status")
    
    def upload(self, commit_message=None, pull_first=True):
        """
        전체 업로드 프로세스
        
        Args:
            commit_message: 커밋 메시지
            pull_first: push 전에 pull 실행 여부
        
        Returns:
            bool: 성공 여부
        """
        print("="*70)
        print("Git 업로드 시작")
        print("="*70)
        
        # Git 설치 확인
        if not self.check_git_installed():
            return False
        
        # 저장소 초기화 또는 클론
        if not self.init_or_clone():
            return False
        
        # 상태 확인
        self.status()
        
        # Pull
        if pull_first:
            self.pull()
        
        # Add
        self.add_all()
        
        # Commit
        if not self.commit(commit_message):
            return False
        
        # Push
        if not self.push():
            print("\nPush 실패. 재시도 중...")
            # 한 번 더 시도
            if not self.push():
                print("\nPush가 계속 실패합니다.")
                print("수동 해결 방법:")
                print(f"   cd {self.repo_path}")
                print(f"   git remote -v")
                print(f"   git push -u origin {self.branch}")
                return False
        
        print("\n" + "="*70)
        print("업로드 완료!")
        print("="*70)
        
        return True


# ===== 메인 실행 코드 =====
if __name__ == "__main__":
    # 설정
    REPO_PATH = r"C:\Users\IPC\Desktop\mechmind\result\objects_ply_data"
    REMOTE_URL = "git clone https://intertek:a0766efec42d41edbe64702231b6eee5@be.axnexus.net:8080/datasets/intertek/objects_ply_data.git"
    
    # Git 업로더 생성
    uploader = GitUploader(REPO_PATH, REMOTE_URL)
    
    # 업로드 실행
    success = uploader.upload(
        commit_message="object 3D scan Data",
        pull_first=True
    )
    
    if success:
        print("\n모든 작업이 완료되었습니다!")
    else:
        print("\n일부 작업이 실패했습니다. 로그를 확인해주세요.")