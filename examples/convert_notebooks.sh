doc_path=../docs/src
for notebook in $(ls *.ipynb)
do
    filename=$(basename $notebook .ipynb)
    if [ -f $doc_path/tutorials/$filename.md ]; then
        echo "The file '$filename' already exists. Deleting..."
        rm -rf $doc_path/$filename*
    fi
    jupyter nbconvert --to markdown --output-dir $doc_path/tutorials/ $notebook
done
