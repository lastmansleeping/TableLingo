#!/bin/bash
MODELS_DIR="models"
DART_TARGET="data/dart/tgt"
DART_DEPTH="data/dart/depth"

for MODEL in bart t5 robertashare
do
	for VERSION in v0 v1 v2
	do
		for SPLIT in dev test
		do
			if [ -f "${MODELS_DIR}/${MODEL}_${VERSION}/pred/${SPLIT}.txt" ]; then
				python pipeline/evaluate.py \
				--pred "${MODELS_DIR}/${MODEL}_${VERSION}/pred/${SPLIT}.txt" \
				--ref "${DART_TARGET}/${SPLIT}.txt" \
				--depth "${DART_DEPTH}/${SPLIT}.txt"
			else
				echo "!!! ${MODELS_DIR}/${MODEL}_${VERSION}/pred/${SPLIT}.txt does not exist !!!"
			fi
		done
	done
done