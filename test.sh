CUDA_VISIBLE_DEVICES=5 python runner.py -m test -c 0927-183027.15

# 0926-233628.18, dev_ppl: 8.85018, dev_bleu: 0.29988451660666743, test_bleu: 0.27734843053470987

#<<< ./multi-bleu.perl data/test.de-en.en < test_results.0926-233628.18.txt
#>>> BLEU = 27.75, 63.5/36.2/22.4/14.1 (BP=0.950, ratio=0.952, hyp_len=124804, ref_len=131141)
#<<< ./multi-bleu.perl data/valid.de-en.en <  valid_results.0926-233628.18.txt
#<<< BLEU = 29.99, 65.2/38.8/24.8/16.2 (BP=0.945, ratio=0.947, hyp_len=122197, ref_len=129091)

# logs:
# train.0926-174637.log
# train.0926-222415.log
# train.0926-233628.log

# 0927-025924.29, dev_ppl: 8.74217, dev_bleu: 0.29782996690350605, test_bleu: 0.2777250676922146
#>>> ./multi-bleu.perl data/test.de-en.en < test_results.0927-025924.29.txt
#<<< BLEU = 27.78, 64.2/36.7/22.8/14.4 (BP=0.936, ratio=0.938, hyp_len=123050, ref_len=131141)
#>>> ./multi-bleu.perl data/valid.de-en.en < valid_results.0927-025924.29.txt
#<<< BLEU = 29.79, 65.8/39.0/24.9/16.2 (BP=0.935, ratio=0.937, hyp_len=120938, ref_len=129091)

#0927-183027.14 dev_ppl: 8.53508, dev_bleu: 0.3011027888891785, test_bleu: 0.2799802141162938
#>>> ./multi-bleu.perl data/test.de-en.en < test_results.0927-183027.14.txt
#<<< BLEU = 28.01, 64.3/36.8/22.9/14.6 (BP=0.938, ratio=0.940, hyp_len=123283, ref_len=131141)
#>>> ./multi-bleu.perl data/valid.de-en.en < valid_results.0927-183027.14.txt
#<<< BLEU = 30.12, 65.9/39.2/25.1/16.4 (BP=0.939, ratio=0.941, hyp_len=121429, ref_len=129091)


