import numpy as np
import matplotlib.pyplot as pt


def main(exp_index):
    c = 0
    first_neq = 0
    for i in range(1, 407):
        # 1
        stem_img = pt.imread(
            'C:\\Users\\Borui He\\OneDrive - Syracuse University\\falling detection\\camera_images\\env_stem\\standing\\{}.png'.format(2*i-1),
            format='png')
        branch_img = pt.imread(
            'C:\\Users\\Borui He\\OneDrive - Syracuse University\\falling detection\\camera_images\\env_branch\\standing\\{}.png'.format(2*i-1),
            format='png')
        # 2
        none_s_img = pt.imread(
            'C:\\Users\\Borui He\\OneDrive - Syracuse University\\falling detection\\camera_images\\none_s\\standing\\{}.png'.format(2*i-1),
            format='png')
        none_b_img = pt.imread(
            'C:\\Users\\Borui He\\OneDrive - Syracuse University\\falling detection\\camera_images\\none_b\\standing\\{}.png'.format(2*i-1),
            format='png')
        # 3
        FG_stem = pt.imread(
            'C:\\Users\\Borui He\\OneDrive - Syracuse University\\falling detection\\camera_images\\FG_stem\\standing\\{}.png'.format(2*i-1),
            format='png')
        FG_branch = pt.imread(
            'C:\\Users\\Borui He\\OneDrive - Syracuse University\\falling detection\\camera_images\\FG_branch\\standing\\{}.png'.format(2*i-1),
            format='png')

        if exp_index == 1:
            if not ((stem_img-branch_img) == 0).all():
                if c == 0:
                    first_neq = i
                c += 1
            
        elif exp_index == 2:
            if not ((none_s_img-none_b_img) == 0).all():
                if c == 0:
                    first_neq = i
                c += 1

        elif exp_index == 3:
            if not ((FG_stem-FG_branch) == 0).all():
                if c == 0:
                    first_neq = i
                c += 1

    if exp_index == 1:
        print('experiment #1: Apply an external force')
    elif exp_index == 2:
        print('experiment #2: No external force')
    elif exp_index == 3:
        print('experiment #3: No gravity')

    if c == 0:
        print('No NEQ found, {} NEQ in total'.format(c))
    else:
        print('frist NEQ at loop {}, {} NEQ in total'.format(first_neq, c))

main(exp_index=1)
main(exp_index=2)
main(exp_index=3)