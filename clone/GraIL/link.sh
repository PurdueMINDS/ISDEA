#
for i in 1 2; do
    #
    top=../../../../data
    mkdir -p datasets/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/entities.dict datasets/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/relations.dict datasets/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/observe.txt datasets/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/train.txt datasets/FD${i}-trans
    ln -sb ${top}/FD${i}-trans/valid.txt datasets/FD${i}-trans
    mkdir -p datasets/FD${i}-ind
    ln -sb ${top}/FD${i}-ind/entities.dict datasets/FD${i}-ind
    ln -sb ${top}/FD${i}-ind/relations.dict datasets/FD${i}-ind
    ln -sb ${top}/FD${i}-ind/observe.txt datasets/FD${i}-ind
    ln -sb ${top}/FD${i}-ind/test.txt datasets/FD${i}-ind
done

#
for prefix in WN18RR FB237 NELL995; do
    for suffix in 1 2 3 4; do
        #
        top=../../../../data
        data=${prefix}${suffix}
        mkdir -p datasets/${data}-trans
        ln -sb ${top}/${data}-trans/entities.dict datasets/${data}-trans
        ln -sb ${top}/${data}-trans/relations.dict datasets/${data}-trans
        ln -sb ${top}/${data}-trans/train.txt datasets/${data}-trans
        ln -sb ${top}/${data}-trans/valid.txt datasets/${data}-trans
        mkdir -p datasets/${data}-ind
        ln -sb ${top}/${data}-ind/entities.dict datasets/${data}-ind
        ln -sb ${top}/${data}-ind/relations.dict datasets/${data}-ind
        ln -sb ${top}/${data}-ind/observe.txt datasets/${data}-ind
        ln -sb ${top}/${data}-ind/test.txt datasets/${data}-ind
    done
done

#
for prefix in WN18RR FB237 NELL995; do
    for suffix in 1 2 3 4; do
        #
        top=../../../../data
        data=${prefix}${suffix}
        mkdir -p datasets/${data}-perm-ind
        ln -sb ${top}/${data}-ind-perm/entities.dict datasets/${data}-perm-ind
        ln -sb ${top}/${data}-ind-perm/relations.dict datasets/${data}-perm-ind
        ln -sb ${top}/${data}-ind-perm/observe.txt datasets/${data}-perm-ind
        ln -sb ${top}/${data}-ind-perm/test.txt datasets/${data}-perm-ind
    done
done
