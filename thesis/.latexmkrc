$out_dir = '../workspace/thesis_build_artifacts';

# Hook: copy PDF back to source dir after successful build
$latex_silent_switch = '';
$pdflatex_silent_switch = '';

END {
    # $rootfile_name is set by latexmk after determining the root file
    my $pdf_src = "$out_dir/thesis.pdf";
    my $pdf_dst = "thesis.pdf";
    if (-f $pdf_src) {
        use File::Copy;
        copy($pdf_src, $pdf_dst) or warn "Failed to copy PDF: $!";
    }
}
