$pdf_mode = 1;
$out_dir = 'build';

# latexmk runs bibtex with the aux directory as its working directory, so
# references.bib sits one level up and is otherwise not found.
ensure_path('BIBINPUTS', '..');
