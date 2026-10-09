.PHONY: check test verify-fixtures verify convert-com2 document-com2 inspect-cladder inspect-csuite inspect-evidence-inference convert-cladder document-cladder convert-wiqa document-wiqa convert-ccrgb document-ccrgb convert-lower document-lower

check: test verify-fixtures

test:
	PYTHONPYCACHEPREFIX=.pycache python3 -m unittest discover -s tests -v

verify-fixtures:
	python3 src/verify.py --records-dir tests/fixtures --splits-dir tests/fixtures/splits

verify:
	python3 src/verify.py --records-dir data/records --splits-dir data/splits

convert-com2:
	python3 src/convert_com2.py

document-com2:
	python3 src/render_com2_card.py

inspect-cladder:
	python3 src/inspect_cladder.py

inspect-csuite:
	python3 src/inspect_csuite.py

inspect-evidence-inference:
	python3 src/inspect_evidence_inference.py

convert-cladder:
	python3 src/convert_cladder.py

document-cladder:
	python3 src/render_cladder_card.py

convert-wiqa:
	python3 src/convert_wiqa.py

document-wiqa:
	python3 src/render_wiqa_card.py

convert-ccrgb:
	@test -n "$(CCRGB_RAW)" || (echo "CCRGB_RAW=/absolute/path/to/ccrgb_6000_worlds.jsonl is required" && exit 2)
	python3 src/convert_ccrgb.py "$(CCRGB_RAW)"

document-ccrgb:
	@test -n "$(CCRGB_RAW)" || (echo "CCRGB_RAW=/absolute/path/to/ccrgb_6000_worlds.jsonl is required" && exit 2)
	python3 src/render_ccrgb_card.py --raw "$(CCRGB_RAW)"

convert-lower:
	python3 src/convert_counterbench.py
	python3 src/convert_causalt5k.py
	python3 src/convert_corr2cause.py
	python3 src/convert_crass.py
	python3 src/convert_meter.py
	python3 src/convert_pubmedcausal.py

document-lower:
	python3 src/render_lower_priority_cards.py
