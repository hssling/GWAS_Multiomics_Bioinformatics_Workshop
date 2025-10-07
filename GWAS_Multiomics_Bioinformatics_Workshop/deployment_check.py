#!/usr/bin/env python3
"""
GWAS, Multiomics & Bioinformatics Workshop - Deployment Status Check

This script checks the deployment status of the workshop components
and provides detailed feedback on repository health.

Author: Dr. Siddalingaiah H S
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import json

class DeploymentChecker:
    def __init__(self, github_username="hssling", repo_name="GWAS_Multiomics_Bioinformatics_Workshop"):
        self.github_username = github_username
        self.repo_name = repo_name
        self.repo_url = f"https://github.com/{github_username}/{repo_name}"
        self.pages_url = f"https://{github_username}.github.io/{repo_name}"
        self.streamlit_url = f"https://{repo_name.replace('_', '-')}-{github_username}.streamlit.app"

        print("🔍 GWAS & Multiomics Bioinformatics Workshop - Deployment Checker")
        print("=" * 70)
        print(f"Repository: {self.repo_url}")
        print(f"Pages URL:  {self.pages_url}")
        print(f"Streamlit:   {self.streamlit_url}")
        print()

    def check_local_repository(self):
        """Check local Git repository status"""
        print("📁 Local Repository Status:")
        print("-" * 30)

        try:
            # Check if Git repository
            result = subprocess.run(["git", "status"], capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                print("✅ Git repository initialized")

                # Check for uncommitted changes
                if "nothing to commit" in result.stdout:
                    print("✅ All changes committed")
                else:
                    print("⚠️  Uncommitted changes detected")
                    print("   Run: git add . && git commit -m \"update\"")

                # Check remote
                remote_result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
                if self.repo_url in remote_result.stdout:
                    print("✅ GitHub remote configured")
                else:
                    print("❌ GitHub remote not configured")
                    print(f"   Run: git remote add origin {self.repo_url}")

            else:
                print("❌ Not a Git repository")
                print("   Run: git init")

        except FileNotFoundError:
            print("❌ Git not installed or not in PATH")

    def check_file_structure(self):
        """Check that all required files are present"""
        print("\n📂 Repository Structure Check:")
        print("-" * 35)

        required_files = [
            "README.md",
            "app.py",
            "requirements.txt",
            "LICENSE",
            ".gitignore",
            ".streamlit/config.toml",
            ".github/workflows/ci-cd.yml",
            ".github/workflows/pages.yml"
        ]

        required_dirs = [
            "Sessions",
            "Data",
            "Exercises",
            "Quizzes",
            "Scripts"
        ]

        # Check files
        all_files_good = True
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
                all_files_good = False

        # Check directories
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                # Count files in directory
                file_count = len(list(Path(dir_path).rglob("*"))) - len(list(Path(dir_path).rglob("*/")))
                print(f"✅ {dir_path}/ ({file_count} files)")
            else:
                print(f"❌ Missing: {dir_path}/")
                all_files_good = False

        if all_files_good:
            print("\n✅ All required files and directories present")
        else:
            print("\n❌ Some files/directories missing - run the file creation commands again")

    def check_github_repository(self):
        """Check GitHub repository status via API"""
        print("\n🐙 GitHub Repository Check:")
        print("-" * 28)

        try:
            # Check repository exists
            api_url = f"https://api.github.com/repos/{self.github_username}/{self.repo_name}"
            response = requests.get(api_url)

            if response.status_code == 200:
                repo_data = response.json()
                print("✅ Repository exists on GitHub")
                print(f"   Stars: {repo_data.get('stargazers_count', 0)}")
                print(f"   Forks: {repo_data.get('forks_count', 0)}")
                print(f"   Size: {repo_data.get('size', 0)} KB")

                # Check if public
                if repo_data.get('private', True):
                    print("❌ Repository is private")
                    print("   Make it public in repository Settings → Danger Zone")
                else:
                    print("✅ Repository is public")

                # Check topics
                topics = repo_data.get('topics', [])
                if 'bioinformatics' in topics and 'genomics' in topics:
                    print("✅ Repository topics configured")
                else:
                    print("⚠️  Repository topics not fully configured")
                    print("   Add topics: bioinformatics, genomics, multiomics, gwas")

            elif response.status_code == 404:
                print("❌ Repository does not exist on GitHub")
                print("   Create repository at: https://github.com/new")
                print(f"   Name: {self.repo_name}")
            else:
                print(f"❌ API error: HTTP {response.status_code}")

        except requests.RequestException as e:
            print(f"❌ Cannot connect to GitHub API: {e}")
            print("   Check internet connection")

    def check_github_pages(self):
        """Check GitHub Pages deployment"""
        print("\n📄 GitHub Pages Check:")
        print("-" * 24)

        try:
            response = requests.get(self.pages_url)
            if response.status_code == 200:
                print("✅ GitHub Pages deployed successfully")
                print(f"   URL: {self.pages_url}")

                # Check if it contains workshop content
                if "GWAS" in response.text and "Bioinformatics" in response.text:
                    print("✅ Pages contain workshop documentation")
                else:
                    print("⚠️  Pages content may not be properly generated")

            elif response.status_code == 404:
                print("❌ GitHub Pages not deployed")
                print("   Check: Repository Settings → Pages")
                print("   Source: Deploy from a branch")
                print("   Branch: main, Folder: docs/")
            else:
                print(f"❌ Pages error: HTTP {response.status_code}")

        except requests.RequestException as e:
            print(f"❌ Cannot access Pages URL: {e}")

    def check_streamlit_deployment(self):
        """Check Streamlit deployment"""
        print("\n🖥️ Streamlit Deployment Check:")
        print("-" * 32)

        try:
            response = requests.get(self.streamlit_url)
            if response.status_code == 200:
                print("✅ Streamlit app deployed successfully")
                print(f"   URL: {self.streamlit_url}")

                # Check if workshop app is loading
                if "GWAS" in response.text:
                    print("✅ Workshop application running")
                else:
                    print("⚠️  App may not be loading correctly")

            elif response.status_code in [404, 503]:
                print("❌ Streamlit app not deployed")
                print("   Deploy at: https://share.streamlit.io")
                print("   Connect repository and set main file path: app.py")
            else:
                print(f"❌ Deployment error: HTTP {response.status_code}")

        except requests.RequestException as e:
            print(f"❌ Cannot access Streamlit URL: {e}")

    def check_github_actions(self):
        """Check GitHub Actions status"""
        print("\n⚙️ GitHub Actions CI/CD Check:")
        print("-" * 32)

        try:
            api_url = f"https://api.github.com/repos/{self.github_username}/{self.repo_name}/actions/runs"
            response = requests.get(api_url)

            if response.status_code == 200:
                runs_data = response.json()
                runs = runs_data.get('workflow_runs', [])

                if runs:
                    latest_run = runs[0]  # Most recent run
                    status = latest_run.get('status', 'unknown')
                    conclusion = latest_run.get('conclusion', 'unknown')

                    if status == 'completed' and conclusion == 'success':
                        print("✅ Latest GitHub Actions run successful")
                    elif status == 'completed' and conclusion == 'failure':
                        print("❌ Latest GitHub Actions run failed")
                        print("   Check: Repository Actions tab for details")
                    elif status == 'in_progress':
                        print("⏳ GitHub Actions currently running")
                    else:
                        print(f"⚠️  Actions status: {status}/{conclusion}")
                else:
                    print("❌ No GitHub Actions runs found")
                    print("   Actions should run automatically on push")

            else:
                print(f"❌ Cannot check Actions status: HTTP {response.status_code}")

        except requests.RequestException as e:
            print(f"❌ Cannot connect to GitHub API: {e}")

    def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        print("\n📋 DEPLOYMENT STATUS REPORT")
        print("=" * 70)

        self.check_local_repository()
        self.check_file_structure()
        self.check_github_repository()
        self.check_github_pages()
        self.check_streamlit_deployment()
        self.check_github_actions()

        print("\n🎯 DEPLOYMENT RECOMMENDATIONS")
        print("=" * 70)
        print("If any checks failed above, follow these steps:")
        print()
        print("1. 🐙 GitHub Repository:")
        print("   - Create at: https://github.com/new")
        print(f"   - Name: {self.repo_name}")
        print("   - Make public, no README initialization")
        print()
        print("2. 📤 Push Code:")
        print("   - Run: python deployment_check.py (to see git status)")
        print("   - Run the commands in git_push_commands.txt")
        print()
        print("3. ⚙️ Enable Features:")
        print("   - GitHub Pages: Settings → Pages → Deploy from docs/")
        print("   - Streamlit: share.streamlit.io → Connect repository")
        print("   - Topics: Add bioinformatics, genomics, gwas tags")
        print()
        print("4. 🔍 Verify Deployment:")
        print(f"   - Repository: {self.repo_url}")
        print(f"   - Pages: {self.pages_url}")
        print("   - Streamlit: Check share.streamlit.io for your deployment")
        print()
        print("🎉 Target URLs after successful deployment:")
        print(f"   📖 Repository: {self.repo_url}")
        print(f"   📄 Documentation: {self.pages_url}")
        print("   🖥️ Interactive App: [Check Streamlit Cloud after deployment]")

def main():
    """Main deployment check function"""
    if len(sys.argv) > 1:
        username = sys.argv[1]
        repo_name = sys.argv[2] if len(sys.argv) > 2 else "GWAS_Multiomics_Bioinformatics_Workshop"
    else:
        username = "hssling"
        repo_name = "GWAS_Multiomics_Bioinformatics_Workshop"

    checker = DeploymentChecker(username, repo_name)
    checker.generate_deployment_report()

if __name__ == "__main__":
    main()
