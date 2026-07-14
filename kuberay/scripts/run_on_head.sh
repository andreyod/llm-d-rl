#!/usr/bin/env bash
# Launch a training run on the RayCluster head from your laptop.
#
# Resolves the head pod by its Ray label (namespace from deploy/deploy.env),
# copies run_test.sh into it, and runs it there. All arguments except this
# script's own --fg flag are passed straight through to run_test.sh.
#
# Usage:
#   scripts/run_on_head.sh --mode epp                 # background on pod + tail the log
#   scripts/run_on_head.sh --mode epp --steps 20 --tp 2
#   scripts/run_on_head.sh --fg --mode native         # run attached (foreground)
#
# Modes / options are run_test.sh's: --mode native|epp, --steps, --tp, --n, --name, --reqlog.
#
# Execution model:
#   default  - nohup run_test.sh on the head into /tmp/train.log, then tail -f it.
#              The run survives a laptop disconnect; Ctrl-C only detaches the tail.
#   --fg     - run attached; output streams live, but dropping the connection
#              kills the run. Fine for short interactive tests.
#
# Requires: kubectl on PATH with a valid context (set KUBECONFIG as needed) and
# the target namespace exported: export NAMESPACE=<your-namespace>.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_LOG="/tmp/train.log"

# Namespace is per-user and comes from the environment. Mandatory, no default.
NS="${NAMESPACE:?NAMESPACE not set - export NAMESPACE=<your-namespace>}"

# Split out our own --fg flag; everything else is forwarded to run_test.sh.
FG=0
PASS=()
for arg in "$@"; do
  case "$arg" in
    --fg) FG=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) PASS+=("$arg") ;;
  esac
done

HEAD="$(kubectl get pod -n "$NS" -l ray.io/node-type=head \
  -o jsonpath='{.items[0].metadata.name}')"
[ -n "$HEAD" ] || { echo "ERROR: no head pod (label ray.io/node-type=head) in namespace $NS" >&2; exit 1; }
echo "==> head pod: $HEAD (namespace $NS)"

echo "==> copying run_test.sh to $HEAD:/tmp/run_test.sh"
kubectl cp "$SCRIPT_DIR/run_test.sh" "$NS/$HEAD:/tmp/run_test.sh"

if [ "$FG" -eq 1 ]; then
  echo "==> running attached (foreground): run_test.sh ${PASS[*]}"
  exec kubectl exec -it -n "$NS" "$HEAD" -- bash /tmp/run_test.sh "${PASS[@]}"
fi

echo "==> launching in background on the pod: run_test.sh ${PASS[*]}"
kubectl exec -n "$NS" "$HEAD" -- bash -c \
  'nohup bash /tmp/run_test.sh "$@" > "'"$REMOTE_LOG"'" 2>&1 & echo "launched pid $!"' \
  _ "${PASS[@]}"

cat <<EOF
==> streaming $REMOTE_LOG (Ctrl-C detaches the tail; the run keeps going on the pod)
    reattach later:  kubectl exec -n $NS $HEAD -- tail -f $REMOTE_LOG
    stop the run:    kubectl exec -n $NS $HEAD -- pkill -f main_ppo
EOF
exec kubectl exec -n "$NS" "$HEAD" -- tail -f "$REMOTE_LOG"
