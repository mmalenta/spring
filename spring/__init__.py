from datetime import datetime

with open(".git/logs/HEAD") as gf:
  all_lines = gf.read().splitlines()
  last_commit = all_lines[-1]

# Remove the commit message
last_commit = last_commit.split("\t")[0]

git_sha = last_commit.split()[1][:7]
# Read from the end - full name is unreliable as it may not use
# just two words (it can be anything really)
date = datetime.utcfromtimestamp(int(last_commit.split()[-2])).strftime("%y%m%d")

base_version = "0.4.0"

__version__ = base_version + "+" + date + "." + git_sha