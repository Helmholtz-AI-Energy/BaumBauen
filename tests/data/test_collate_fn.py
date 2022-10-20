import torch as t
import unittest

from baumbauen.utils import pad_collate_fn, rel_pad_collate_fn, construct_rel_recvs, construct_rel_sends


class CollateFnTest(unittest.TestCase):
    def test_pad_collate_fn(self):
        '''
        '''


        x_1 = t.tensor([
            #f1 f2 f3 f4
            [1, 2, 3, 4], # a
            [4, 3, 2, 1], # b
        ])
        y_1 = t.tensor([
            #a  b
            [0, 1], # a
            [1, 0], # b
        ])
        x_2 = t.tensor([
            #f1 f2 f3 f4
            [1, 2, 3, 4], # a
            [4, 3, 2, 1], # b
            [4, 3, 3, 4], # c
        ])
        y_2 = t.tensor([
            #a  b  c
            [0, 1, 2], # a
            [1, 0, 2], # b
            [2, 2, 0], # c
        ])

        # Note that data is padded with zeros
        comparison_data = t.tensor([
            [
                #f1 f2 f3 f4
                [1, 2, 3, 4], # a
                [4, 3, 2, 1], # b
                [0, 0, 0, 0], # padding
            ],
            [
                #f1 f2 f3 f4
                [1, 2, 3, 4], # a
                [4, 3, 2, 1], # b
                [4, 3, 3, 4], # c
            ],
        ])

        # to match pad_collate_fn ordering
        comparison_data = comparison_data.transpose(0, 1)  # (l, n, f)

        # Note that targets are padded with -1
        comparison_target = t.tensor([
            [
                #a  b  padding
                [ 0,  1, -1], # a
                [ 1,  0, -1], # b
                [-1, -1, -1], # padding
            ],
            [
                #a  b  c
                [0, 1, 2], # a
                [1, 0, 2], # b
                [2, 2, 0], # c
            ],
        ])

        batch = [(x_1, y_1), (x_2, y_2)]

        data, target = pad_collate_fn(batch)

        self.assertTrue((data == comparison_data).all())
        self.assertTrue((target == comparison_target).all())

    def test_rel_pad_collate_fn_bs1(self):

        x_2 = t.tensor([
            #f1 f2 f3 f4
            [1, 2, 3, 4], # a
            [4, 3, 2, 1], # b
            [4, 3, 3, 4], # c
        ])
        y_2 = t.tensor([
            #a  b  c
            [0, 1, 2], # a
            [1, 0, 2], # b
            [2, 2, 0], # c
        ])

        comparison_data = t.tensor([
            [
                #f1 f2 f3 f4
                [1, 2, 3, 4], # a
                [4, 3, 2, 1], # b
                [4, 3, 3, 4], # c
            ],
        ])

        # to match pad_collate_fn ordering
        comparison_data = comparison_data.transpose(0,1)  # (l, n, f)
        comparison_target = t.tensor([
            [
                #a  b  c
                [0, 1, 2], # a
                [1, 0, 2], # b
                [2, 2, 0], # c
            ],
        ])

        batch = [(x_2, y_2),]
        data, target = rel_pad_collate_fn(batch)
        data, rel_recvs, rel_sends = data

        comparison_rel_recv = t.tensor([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 0.]
            ])

        comparison_rel_send = t.tensor([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            ])

        self.assertTrue((rel_recvs[0] == comparison_rel_recv).all())
        self.assertTrue((rel_sends[0] == comparison_rel_send).all())

        self.assertTrue((data == comparison_data).all())
        self.assertTrue((target == comparison_target).all())

    def test_rel_pad_collate_fn(self):
        '''
        Tests the padding with rel_rec and rel_send for NRI models works as expected
        '''

        x_1 = t.tensor([
            #f1 f2 f3 f4
            [1, 2, 3, 4], # a
            [4, 3, 2, 1], # b
        ])
        y_1 = t.tensor([
            #a  b
            [0, 1], # a
            [1, 0], # b
        ])
        x_2 = t.tensor([
            #f1 f2 f3 f4
            [1, 2, 3, 4], # a
            [4, 3, 2, 1], # b
            [4, 3, 3, 4], # c
        ])
        y_2 = t.tensor([
            #a  b  c
            [0, 1, 2], # a
            [1, 0, 2], # b
            [2, 2, 0], # c
        ])

        comparison_data = t.tensor([
            [
                #f1 f2 f3 f4
                [1, 2, 3, 4], # a
                [4, 3, 2, 1], # b
                [0, 0, 0, 0], # padding
            ],
            [
                #f1 f2 f3 f4
                [1, 2, 3, 4], # a
                [4, 3, 2, 1], # b
                [4, 3, 3, 4], # c
            ],
        ])

        # to match pad_collate_fn ordering
        comparison_data = comparison_data.transpose(0, 1)  # (l, n, f)
        comparison_target = t.tensor([
            [
                # a    b  padding
                [ 0,  1, -1],  # a
                [ 1,  0, -1],  # b
                [-1, -1, -1],  # padding
            ],
            [
                #a  b  c
                [0, 1, 2], # a
                [1, 0, 2], # b
                [2, 2, 0], # c
            ],
        ])

        batch = [(x_1, y_1), (x_2, y_2)]
        comparison_rel_recvs = t.tensor([
            [
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]
            ],
            [
                [0., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 0., 1.],
                [0., 0., 0.],
            ],
        ])

        comparison_rel_sends = t.tensor([
            [
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]
            ],
            [
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.],
                [0., 0., 0.],
                [0., 0., 1.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
            ],
        ])

        (data, rel_recvs, rel_sends), target = rel_pad_collate_fn(batch)

        self.assertTrue((data == comparison_data).all())
        self.assertTrue((target == comparison_target).all())
        self.assertTrue((rel_sends == comparison_rel_sends).all())
        self.assertTrue((rel_recvs == comparison_rel_recvs).all())

    def test_construct_rel_recvs_self(self):
        nl2pad2 = t.tensor([
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [0., 1.]
            ])
        nl2pad3 =  t.tensor([
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
            ])
        nl2pad4 =  t.tensor([
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
            ])
        nl3pad3 =  t.tensor([
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.]
            ])
        nl3pad4 =  t.tensor([
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
            ])
        nl4pad4 =  t.tensor([
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.]
            ])

        comparison_rel_recvs = {
                (2, 2) : nl2pad2,
                (2, 3) : nl2pad3,
                (2, 4) : nl2pad4,
                (3, 3) : nl3pad3,
                (3, 4) : nl3pad4,
                (4, 4) : nl4pad4,
                }
        for bs in [1, 2, 3, 4]:
            lens = tuple(t.randint(2, 5, (bs,)))
            pad_len = max(lens)
            rel_recvs = construct_rel_recvs(lens, self_interaction=True)
            comparison = t.stack([comparison_rel_recvs[(l.item(), pad_len.item())] for l in lens])
            self.assertTrue((rel_recvs == comparison).all())


    def test_construct_rel_recvs(self):
        nl2pad2 = t.tensor([
            [0., 0.],
            [1., 0.],
            [0., 1.],
            [0., 0.]
            ])
        nl2pad3 =  t.tensor([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
            ])
        nl2pad4 =  t.tensor([
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
            ])
        nl3pad3 =  t.tensor([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 0.]
            ])
        nl3pad4 =  t.tensor([
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
            ])
        nl4pad4 =  t.tensor([
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.]
            ])

        comparison_rel_recvs = {
                (2, 2) : nl2pad2,
                (2, 3) : nl2pad3,
                (2, 4) : nl2pad4,
                (3, 3) : nl3pad3,
                (3, 4) : nl3pad4,
                (4, 4) : nl4pad4,
                }
        for bs in [1, 2, 3, 4]:
            lens = tuple(t.randint(2, 5, (bs,)))
            pad_len = max(lens)
            rel_recvs = construct_rel_recvs(lens, self_interaction=False)
            comparison = t.stack([comparison_rel_recvs[(l.item(), pad_len.item())] for l in lens])
            self.assertTrue((rel_recvs == comparison).all())

    def test_construct_rel_sends_self(self):
        nl2pad2 = t.tensor([
            [1., 0.],
            [0., 1.],
            [1., 0.],
            [0., 1.],
            ])
        nl2pad3 =  t.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
            ])
        nl2pad4 =  t.tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
            ])
        nl3pad3 =  t.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            ])
        nl3pad4 =  t.tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
            ])
        nl4pad4 =  t.tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            ])
        comparison_rel_sends = {
                (2, 2) : nl2pad2,
                (2, 3) : nl2pad3,
                (2, 4) : nl2pad4,
                (3, 3) : nl3pad3,
                (3, 4) : nl3pad4,
                (4, 4) : nl4pad4,
                }
        for bs in [1, 2, 3, 4]:
            lens = tuple(t.randint(2, 5, (bs,)))
            pad_len = max(lens)
            rel_sends = construct_rel_sends(lens, self_interaction=True)
            comparison = t.stack([comparison_rel_sends[(l.item(), pad_len.item())] for l in lens])
            self.assertTrue((rel_sends == comparison).all())


    def test_construct_rel_sends(self):
        nl2pad2 = t.tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [0., 0.],
            ])
        nl2pad3 =  t.tensor([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
            ])
        nl2pad4 =  t.tensor([
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
            ])
        nl3pad3 =  t.tensor([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            ])
        nl3pad4 =  t.tensor([
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
            ])
        nl4pad4 =  t.tensor([
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            ])
        comparison_rel_sends = {
                (2, 2) : nl2pad2,
                (2, 3) : nl2pad3,
                (2, 4) : nl2pad4,
                (3, 3) : nl3pad3,
                (3, 4) : nl3pad4,
                (4, 4) : nl4pad4,
                }
        for bs in [1, 2, 3, 4]:
            lens = tuple(t.randint(2, 5, (bs,)))
            pad_len = max(lens)
            rel_sends = construct_rel_sends(lens, self_interaction=False)
            comparison = t.stack([comparison_rel_sends[(l.item(), pad_len.item())] for l in lens])
            self.assertTrue((rel_sends == comparison).all())

if __name__ == '__main__':
    unittest.main()
