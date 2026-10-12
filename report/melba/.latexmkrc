$pdf_mode = 1;
$out_dir = '../build/melba';

# sections/, figures/ and references.bib are shared with the arXiv build one level up.
# bibtex runs inside the out dir, two levels below report/.
ensure_path('TEXINPUTS', '..');
ensure_path('BIBINPUTS', '../..');
