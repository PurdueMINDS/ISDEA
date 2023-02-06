#
for i in 1 2; do
    #
    top=../../../../data
    mkdir -p data/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/entities.dict data/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/relations.dict data/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/observe.txt data/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/train.txt data/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/valid.txt data/FD${i}-trans
    mkdir -p data/FD${i}-ind
    ln -sb ${top}/FD${i}-ind/entities.dict data/FD${i}-ind
    ln -sb ${top}/FD${i}-ind/relations.dict data/FD${i}-ind
    ln -sb ${top}/FD${i}-ind/observe.txt data/FD${i}-ind
    ln -sb ${top}/FD${i}-ind/test.txt data/FD${i}-ind
done

#
for data in WN18RR1 NELL9951; do
    #
    top=../../../../data
    mkdir -p data/${data}-trans
    ln -sb ${top}/${data}-trans/entities.dict data/${data}-trans
    ln -sb ${top}/${data}-trans/relations.dict data/${data}-trans
    ln -sb ${top}/${data}-trans/train.txt data/${data}-trans
    ln -sb ${top}/${data}-trans/valid.txt data/${data}-trans
    mkdir -p data/${data}-ind
    ln -sb ${top}/${data}-ind/entities.dict data/${data}-ind
    ln -sb ${top}/${data}-ind/relations.dict data/${data}-ind
    ln -sb ${top}/${data}-ind/observe.txt data/${data}-ind
    ln -sb ${top}/${data}-ind/test.txt data/${data}-ind
done

#
for data in WN18RR1 NELL9951; do
    #
    top=../../../../data
    mkdir -p data/${data}-perm-ind
    ln -sb ${top}/${data}-ind-perm/entities.dict data/${data}-perm-ind
    ln -sb ${top}/${data}-ind-perm/relations.dict data/${data}-perm-ind
    ln -sb ${top}/${data}-ind-perm/observe.txt data/${data}-perm-ind
    ln -sb ${top}/${data}-ind-perm/test.txt data/${data}-perm-ind
done
