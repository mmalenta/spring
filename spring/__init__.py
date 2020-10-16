from datetime import datetime

with open(".git/logs/HEAD") as gf:
    all_lines = gf.read().splitlines()
    last_commit = all_lines[-1]

git_sha = last_commit.split()[1][:7]
date = datetime.utcfromtimestamp(int(last_commit.split()[5])).strftime("%y%m%d")

__version__ = "0.1.0+" + date + "." + git_sha