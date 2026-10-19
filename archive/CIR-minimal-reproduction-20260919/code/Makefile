PYTHON ?= python

.PHONY: check-all check-dress check-shirt check-toptee train-all train-dress train-shirt train-toptee aggregate-fashioniq compile

check-all:
	$(PYTHON) scripts/check_dataset.py --dataset all

check-dress:
	$(PYTHON) scripts/check_dataset.py --dataset dress

check-shirt:
	$(PYTHON) scripts/check_dataset.py --dataset shirt

check-toptee:
	$(PYTHON) scripts/check_dataset.py --dataset toptee

train-all:
	bash scripts/run_all_fashioniq_stage1.sh

train-dress:
	bash scripts/run_fashioniq_stage1.sh dress

train-shirt:
	bash scripts/run_fashioniq_stage1.sh shirt

train-toptee:
	bash scripts/run_fashioniq_stage1.sh toptee

aggregate-fashioniq:
	$(PYTHON) scripts/aggregate_fashioniq_metrics.py \
		--dress checkpoints/fashioniq/dress/dress_stage1_metrics_best.json \
		--shirt checkpoints/fashioniq/shirt/shirt_stage1_metrics_best.json \
		--toptee checkpoints/fashioniq/toptee/toptee_stage1_metrics_best.json

compile:
	$(PYTHON) -m py_compile src/*.py scripts/*.py
