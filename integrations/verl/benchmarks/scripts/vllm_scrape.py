import time, urllib.request, datetime
import yaml
OUT = "/tmp/vllm_metrics.csv"
EP  = "/tmp/epp-endpoints.yaml"
WANT = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
]
def scalar(text, name):
    tot = None
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                tot = (tot or 0.0) + float(line.rsplit(" ", 1)[1])
            except Exception:
                pass
    return tot
with open(OUT, "a", buffering=1) as f:
    f.write("# ts_iso,epoch,replica,addr," + ",".join(WANT) + "\n")
    while True:
        try:
            eps = yaml.safe_load(open(EP)).get("endpoints", [])
        except Exception:
            eps = []
        now = datetime.datetime.now()
        ts = now.isoformat(timespec="milliseconds")
        ep_sec = now.timestamp()
        for e in eps:
            addr = f'{e["address"]}:{e["port"]}'
            name = e.get("name", addr)
            try:
                txt = urllib.request.urlopen(f"http://{addr}/metrics", timeout=2).read().decode()
                vals = [scalar(txt, m) for m in WANT]
                f.write(f"{ts},{ep_sec:.3f},{name},{addr}," + ",".join("" if v is None else f"{v:.3f}" for v in vals) + "\n")
            except Exception as ex:
                f.write(f"{ts},{ep_sec:.3f},{name},{addr},ERR:{type(ex).__name__}\n")
        time.sleep(1.5)
