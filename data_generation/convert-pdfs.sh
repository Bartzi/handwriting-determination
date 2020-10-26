 #!/bin/bash
 
PDF_LOC="./base_data/raw_documents/*"
PNG_LOC="./base_data/documents"


for f in $PDF_LOC; do mv "$f" "${f// /_}"; done

mkdir -p $PNG_LOC
rm -r $PNG_LOC/*
for d in $PDF_LOC; do
    bname="$(basename $d)"
    filename="${bname%.*}"
    echo "Converting $bname ..."
    convert -density 300 $d -quality 100 -alpha off "$PNG_LOC/$filename.png"
done
