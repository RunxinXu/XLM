from tqdm import tqdm

def pick_out_test():
    test_standford = set()
    test_tiktok = set()
    with open('/mnt/cephfs_new_wj/bytetrans/runxindidi/XLM/mymodule/en2vi/test/standford.en') as fe, \
        open('/mnt/cephfs_new_wj/bytetrans/runxindidi/XLM/mymodule/en2vi/test/standford.vi') as fv:
        for (en, vi) in zip(fe, fv):
            en = en.strip().replace(' ', '')
            vi = vi.strip().replace(' ', '')
            test_standford.add(en+'<!!!!!>'+vi)
    with open('/mnt/cephfs_new_wj/bytetrans/runxindidi/XLM/mymodule/en2vi/test/tiktok.en') as fe, \
        open('/mnt/cephfs_new_wj/bytetrans/runxindidi/XLM/mymodule/en2vi/test/tiktok.vi') as fv:
        for (en, vi) in zip(fe, fv):
            en = en.strip().replace(' ', '')
            vi = vi.strip().replace(' ', '')
            test_tiktok.add(en+'<!!!!!>'+vi)
    print(len(test_standford))
    print(len(test_tiktok))
    with open('/mnt/cephfs_new_wj/bytetrans/runxindidi/XLM/mymodule/en2vi/alldata_without_test/alldata_result') as f, \
        open('/mnt/cephfs_new_wj/bytetrans/runxindidi/XLM/mymodule/en2vi/alldata_without_test/alldata_result_without_test', 'w') as g:
        for line in f:
            line = line.strip()
            line_split = line.split('\t')
            assert len(line_split) == 3
            a = line_split[0].strip().replace(' ', '') + '<!!!!!>' + line_split[1].strip().replace(' ', '')
            count = 0
            if a in test_standford or a in test_tiktok:
                print(a)
                print('find it')
                count += 1
            else:
                g.write(line+'\n')
    print(count)

pick_out_test()
