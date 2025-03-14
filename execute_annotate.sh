# python -m scripts.annotate_pairs_ult --annotator_type random \
#                                  --directory og_dataset \
#                                  --custom_annotator_string random_annotate_test \
#                                  --n_chunks 1 \
#                                  --chunk_number 0 \
#                                  --ignore_existing 1

python -m scripts.annotate_pairs_ult --annotator_type llama3.2_3B \
                                 --directory og_dataset \
                                 --goal_key discoverer \
                                 --custom_annotator_string 3B_annotate_test \
                                 --n_chunks 1 \
                                 --chunk_number 0 \
                                 --ignore_existing 1