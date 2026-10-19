import subprocess
import tempfile
import os

tpath = os.path.join(tempfile.gettempdir(), "gitcred.txt")
with open(tpath, "w", encoding="utf-8") as f:
    f.write("protocol=https\nhost=github.com\n\n")

p = subprocess.run("git credential fill", input=open(tpath, "rb").read(), stdout=subprocess.PIPE, shell=True)
if os.path.exists(tpath):
    os.remove(tpath)

token = [l[9:] for l in p.stdout.decode().splitlines() if l.startswith("password=")][0]

subprocess.run("git add .", shell=True, check=True)
subprocess.run('git commit -m "feat(course-project): Add complete Wokwi simulation, training pipeline, INT8 C-header artifacts, and academic REPORT.md"', shell=True, check=True)
subprocess.run(f"git push https://{token}@github.com/Sais-gallego/developing-aiot.git main", shell=True, check=True)
print("Successfully pushed to GitHub!")
