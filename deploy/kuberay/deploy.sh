#!/usr/bin/env bash
# Deploy (or tear down) the KubeRay example using the single config in
# deploy.env. Renders ray-cluster.yaml.tmpl with the image refs and namespace
# from deploy.env, builds the llmd-epp-configs ConfigMap from the standalone
# EPP config files, and applies both - so the namespace and images are defined
# in exactly one place (deploy.env).
#
# Usage:
#   ./deploy.sh                  # create ConfigMap + apply the cluster
#   ./deploy.sh apply            # same
#   ./deploy.sh delete           # delete the cluster (leaves the ConfigMap)
#   ./deploy.sh configmap        # (re)create the ConfigMap only
#   ./deploy.sh render           # print the rendered manifest to stdout (no kubectl)
#   ./deploy.sh retriever        # apply the BM25 searchr1 retriever (Deployment+Service)
#   ./deploy.sh retriever-delete # delete the retriever
#   ./deploy.sh render-retriever # print the rendered retriever manifest (no kubectl)
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
  envsubst '${NAMESPACE} ${IMG_VERL} ${IMG_CRANE} ${IMG_EPP} ${IMG_ENVOY} ${IMG_SIDECAR}' \
    < ray-cluster.yaml.tmpl
}

render_retriever() {
  # BM25 retriever Deployment+Service (searchr1 workload only). Scoped var list so the
  # $-quoted init-container script is left untouched.
  envsubst '${NAMESPACE} ${IMG_RETRIEVER}' < ../../workloads/searchr1/retriever/retriever.yaml.tmpl
}

create_configmap() {
  # epp-config.yaml / epp-config-pd.yaml / envoy.yaml are the source of truth;
  # build the ConfigMap from them (idempotent apply) into the configured
  # namespace. envoy.yaml is consumed by the llm-d stack (Envoy) integration.
  kubectl create configmap llmd-epp-configs \
    --from-file=epp-config.yaml=../epp-config.yaml \
    --from-file=envoy.yaml=../envoy.yaml \
    --from-file=searchr1_tool_config.yaml=../../workloads/searchr1/tool_config.yaml \
    --namespace "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
}

case "$ACTION" in
  render)            render ;;
  configmap)         create_configmap ;;
  apply)             create_configmap; render | kubectl apply -f - ;;
  delete)            render | kubectl delete -f - ;;
  retriever)         render_retriever | kubectl apply -f - ;;
  retriever-delete)  render_retriever | kubectl delete -f - ;;
  render-retriever)  render_retriever ;;
  *) echo "Unknown action: $ACTION (use apply | delete | configmap | render | retriever | retriever-delete | render-retriever)" >&2; exit 2 ;;
esac
