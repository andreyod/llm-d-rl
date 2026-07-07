#!/usr/bin/env bash
# Deploy (or tear down) the KubeRay example using the single config in
# deploy.env. Renders ray-cluster.yaml.tmpl with the image refs and namespace
# from deploy.env, builds the llmd-epp-configs ConfigMap from the standalone
# EPP config files, and applies both - so the namespace and images are defined
# in exactly one place (deploy.env).
#
# Usage:
#   ./deploy.sh              # create ConfigMap + apply the cluster
#   ./deploy.sh apply        # same
#   ./deploy.sh delete       # delete the cluster (leaves the ConfigMap)
#   ./deploy.sh configmap    # (re)create the ConfigMap only
#   ./deploy.sh render       # print the rendered manifest to stdout (no kubectl)
#
# Requires: envsubst (GNU gettext) and kubectl on PATH.
set -euo pipefail

ACTION="${1:-apply}"
cd "$(dirname "$0")"

# deploy.env provides the IMG_* refs (envsubst reads them from the environment).
set -a
# shellcheck disable=SC1091
. ./deploy.env
set +a

# NAMESPACE is per-user and comes from the environment, not deploy.env. Mandatory,
# no default: :? fails fast (before any kubectl) on empty OR unset.
: "${NAMESPACE:?not set - export NAMESPACE=<your-namespace>}"

render() {
  # Explicit var list keeps envsubst from touching the container-runtime
  # $EPP_IMAGE / $ENVOY_IMAGE in the crane args and the shell $ in postStart.
  envsubst '${NAMESPACE} ${IMG_VERL} ${IMG_CRANE} ${IMG_EPP} ${IMG_ENVOY}' \
    < ray-cluster.yaml.tmpl
}

create_configmap() {
  # epp-config.yaml / epp-config-pd.yaml / envoy.yaml are the source of truth;
  # build the ConfigMap from them (idempotent apply) into the configured
  # namespace. envoy.yaml is consumed by the llm-d stack (Envoy) integration.
  kubectl create configmap llmd-epp-configs \
    --from-file=epp-config.yaml=epp-config.yaml \
    --from-file=envoy.yaml=envoy.yaml \
    --namespace "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
}

case "$ACTION" in
  render)     render ;;
  configmap)  create_configmap ;;
  apply)      create_configmap; render | kubectl apply -f - ;;
  delete)     render | kubectl delete -f - ;;
  *) echo "Unknown action: $ACTION (use apply | delete | configmap | render)" >&2; exit 2 ;;
esac
