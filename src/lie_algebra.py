## This file runs some demos for the Lie Algebra interface

from algebra import semisimple_lie_algebra, representations, tensor_product, embedding, test

if __name__ == '__main__':

    #test.run_unit_test(message_detail=True)

    algebra1 = semisimple_lie_algebra.A(3)
    rep1 = representations.Irrep(algebra1, [1, 2, 4])
    rep2 = representations.Irrep(algebra1, [1, 7, 13])

