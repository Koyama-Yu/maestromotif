# python -m scripts.annotate_pairs_ult --annotator_type random \
#                                  --directory og_dataset \
#                                  --custom_annotator_string random_annotate_test \
#                                  --n_chunks 1 \
#                                  --chunk_number 0 \
#                                  --ignore_existing 1

# python -m scripts.annotate_pairs_ult --annotator_type llama3.2_3B \
#                                  --directory og_dataset \
#                                  --goal_key discoverer \
#                                  --custom_annotator_string 3B_annotate_test \
#                                  --n_chunks 1 \
#                                  --chunk_number 0 \
#                                  --ignore_existing 1

API_KEY="${1:-$FOUNDRY_API_KEY}"
ENDPOINT="${2:-$FOUNDRY_ENDPOINT}"

EXTRA_ARGS=()
if [ -n "$API_KEY" ]; then
  EXTRA_ARGS+=(--api_key "$API_KEY")
fi
if [ -n "$ENDPOINT" ]; then
  EXTRA_ARGS+=(--endpoint "$ENDPOINT")
fi

# python -m scripts.annotate_pairs_ult \
#   --annotator_type llama3.1_70B \
#   --directory og_dataset_copy \
#   --goal_key food \
#   --n_chunks 1 \
#   --chunk_number 0 \
#   --ignore_existing 1 \
#   "${EXTRA_ARGS[@]}"

# python -m scripts.annotate_pairs_ult \
#   --annotator_type llama3.3_70B_foundry \
#   --directory og_dataset_copy \
#   --goal_key food \
#   --n_chunks 1 \
#   --chunk_number 0 \
#   --ignore_existing 1 \
#   --api_key "$API_KEY" \
#   --endpoint "$ENDPOINT" \
#   --deployment_name "Llama-3.3-70B-Instruct"

python -m scripts.annotate_pairs_ult \
  --annotator_type llama3.3_70B_foundry \
  --directory og_dataset_copy \
  --goal_key scroll \
  --n_chunks 1 \
  --chunk_number 0 \
  --ignore_existing 1 \
  --batch_size 1 \
  --max_batches 0 \
  --api_key "$API_KEY" \
  --endpoint "$ENDPOINT" \
  --deployment_name "Llama-3.3-70B-Instruct"


## random
  # python -m scripts.annotate_pairs_ult \
  # --annotator_type random \
  # --directory og_dataset_copy \
  # --goal_key food \
  # --n_chunks 1 \
  # --chunk_number 0 \
  # --ignore_existing 1 \
  # --batch_size 1 \
  # --max_batches 0 \
  # --api_key "$API_KEY" \
  # --endpoint "$ENDPOINT" \
  # --deployment_name "random_test"


