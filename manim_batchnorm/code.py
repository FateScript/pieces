#!/usr/bin/env python3

import numpy as np
import functools
from manim import *

np.random.seed(3)


class Start(ThreeDScene):

    def title(self):
        title = Text("Batch Normalization", font_size=90, color=BLUE)
        info = Text(
            """
            Feng Wang\n
            Github: FateScript\n
            打杂 @ Megvii\n
            """,
            font_size=24,
            color=TEAL,
        )
        VGroup(title, info).arrange(DOWN, center=False, aligned_edge=RIGHT, buff=1)
        self.play(Write(title))
        self.wait(1)
        self.play(Write(info), run_time=2)
        self.wait(24)
        self.play(FadeOut(info, shift=DOWN))
        self.play(title.animate.scale(0.6).to_edge(UP))
        # title.to_edge(UP)
        self.wait(1)
        return title

    def generate_tensor_cube(self, height=3, width=3, channel=3, channel_colors=[YELLOW, BLUE, RED]):
        assert len(channel_colors) == channel

        # channel first
        # make a unit cube
        s1 = Cube(fill_opacity=0.1, stroke_width=4)
        s1.set_color(channel_colors[0])
        s1.set_stroke(WHITE)
        s1.scale(0.5)
        s1.set_opacity(0.5)
        DrawBorderThenFill(s1)

        channel_cubes = [s1]
        for idx, color in enumerate(channel_colors[1:]):
            # Second Cube
            s = s1.copy()
            if idx == 0:
                s.shift(UP, s1.get_arc_length())
            else:
                s.shift(DOWN, s1.get_arc_length())
            s.set_color(color)
            s.set_stroke(WHITE)
            ReplacementTransform(s1.copy(), s)
            channel_cubes.append(s)
        
        # Row 1
        row1 = VGroup(s1, *channel_cubes)

        # Row 2
        row2 = row1.copy()
        row2.set_opacity(0.1)
        row2.shift(IN, channel_cubes[1].get_arc_length())

        ReplacementTransform(row1.copy(), row2)
        row2.set_opacity(0.5)

        # Row 3
        row3 = row1.copy()
        row3.set_opacity(0.1)
        row3.shift(OUT, channel_cubes[1].get_arc_length())

        ReplacementTransform(row1.copy(), row3)
        row3.set_opacity(0.5)
        
        # Matri1
        mat1 = VGroup(row1, row2, row3)

        # Matrix 2
        mat2 = mat1.copy()
        mat2.set_opacity(0.1)
        mat2.shift(RIGHT, channel_cubes[1].get_arc_length())

        ReplacementTransform(mat1.copy(), mat2)
        mat2.set_opacity(0.5)

        # Matrix 3
        mat3 = mat1.copy()
        mat3.set_opacity(0.1)
        mat3.shift(LEFT, channel_cubes[1].get_arc_length())

        ReplacementTransform(mat1.copy(), mat3)
        mat3.set_opacity(0.5)

        # Rotate tensor to help visualizing
        tensor = VGroup(mat1, mat2, mat3).rotate(90 * DEGREES, axis=-Z_AXIS)
        tensor.rotate(20 * DEGREES, axis=X_AXIS).rotate(30 * DEGREES, axis=Y_AXIS).rotate(10 * DEGREES, axis=Z_AXIS)  # noqa

        return tensor

    def generate_2d_tensor(self, value=None, show_create_animate=True):
        if value is None:
            value = [
                [0.1, 1.2, 2.3],
                [3.4, 4.5, 5.6],
                [6.7, 7.8, 8.9],
            ]
        colors = [BLUE, YELLOW, RED]

        m1 = Matrix(value, include_background_rectangle=True)
        m2 = m1.copy()

        shift = 0.5 * LEFT + 0.5 * DOWN
        if show_create_animate:
            matrix_text = Text("Matrix", font_size=80, color=BLUE)
            self.play(Write(matrix_text))
            m1.set_column_colors(BLUE, BLUE, BLUE)
            self.play(ReplacementTransform(matrix_text, m1))
            self.wait()

            m2.set_column_colors(YELLOW, YELLOW, YELLOW)
            self.play(m2.animate.shift(shift), rum_time=2)
            m3 = m2.copy()
            m3.set_column_colors(RED, RED, RED)
            self.play(m3.animate.shift(shift), rum_time=2)
            self.wait(2)

            # arrow logic
            channel_arrow = Arrow(start=RIGHT + UP, end=LEFT + DOWN, color=BLUE)
            channel_text = Text("channel", font_size=40, color=TEAL).next_to(channel_arrow, RIGHT)
            arrow_and_text = VGroup(channel_arrow, channel_text).shift(RIGHT * 2)
            self.play(Write(channel_arrow))
            self.play(Write(channel_text))
            self.wait(2)
            self.play(FadeOut(arrow_and_text))

        else:
            m2.shift(shift)
            m3 = m2.copy()
            m3.shift(shift)

        tensor_group = VGroup(m1, m2, m3)
        if show_create_animate:
            copyed_tensor = tensor_group.copy()
            tensor_text = Text("Tensor", font_size=80, color=BLUE)
            self.play(
                FadeOut(tensor_group),
                ReplacementTransform(copyed_tensor, tensor_text),
            )
            self.wait()
            self.play(ReplacementTransform(tensor_text, tensor_group))

        return tensor_group

    def generate_random_3d_tensor(self, row=3, col=3, channel=3):
        value = np.random.randint(0, 100, size=(channel, row, col)) / 10
        matrix_list = []
        unit_shift = LEFT * 0.5 + DOWN * 0.5
        for i, matrix_value in enumerate(reversed(value)):
            # the last one should be the first in vgroup, so use `reversed`
            matrix = Matrix(matrix_value, fill_opacity=1, include_background_rectangle=True)
            matrix.value = matrix_value
            matrix.shift(unit_shift * i)
            matrix_list.append(matrix)

        return VGroup(*matrix_list).shift((channel - 1) / 2 * -unit_shift)

    def concept_text(self, content=None):
        title = Text("1. Concept", font_size=90, color=BLUE)
        if content is None:
            self.play(Write(title))
        else:
            self.play(ReplacementTransform(content, title))
        self.wait(2)
        self.play(FadeOut(title))

    def batch_concept(self, content):
        # matrix to tensor cube visualize
        self.concept_text(content)

        tensor_matrix = self.generate_2d_tensor()
        self.play(tensor_matrix.animate.shift(3 * LEFT + UP * 0.5).scale(0.5))
        tensor_cube = self.generate_tensor_cube()

        # arrow show up
        arrow_text = Text("batch = 1", color=BLUE).shift(UP)
        arrow = Arrow(start=1.5 * LEFT, end=1.5 * RIGHT)
        self.play(FadeIn(arrow, arrow_text))

        # tensor cube show up
        tensor_cube.scale(0.5).shift(3.5 * RIGHT)
        self.play(FadeIn(tensor_cube))

        # show batch
        tensor_matrix2 = tensor_matrix.copy()
        tensor_cube2 = tensor_cube.copy()
        batch2_arrow_text = Text("batch = 2", color=BLUE).shift(UP)
        self.play(
            ReplacementTransform(arrow_text, batch2_arrow_text),
            tensor_matrix.animate.shift(1.5 * UP),
            tensor_cube.animate.shift(1.2 * UP),
            tensor_matrix2.animate.shift(1 * DOWN),
            tensor_cube2.animate.shift(1.2 * DOWN)
        )
        self.wait(2)

        batch1_text = Text("batch = 1", color=BLUE).shift(UP)
        self.play(
            ReplacementTransform(batch2_arrow_text, batch1_text),
            tensor_matrix.animate.shift(1.5 * DOWN),
            tensor_cube.animate.shift(1.2 * DOWN),
            tensor_matrix2.animate.shift(1 * UP),
            tensor_cube2.animate.shift(1.2 * UP),
        )
        self.play(
            FadeOut(batch1_text),
            FadeOut(tensor_matrix2),
            FadeOut(tensor_cube2),
            FadeOut(tensor_matrix),
            FadeOut(arrow),
        )
        self.wait()
        return tensor_cube

    def display_bn_func(self, input_tensor, scale_back=False):
        self.play(input_tensor.animate.shift(LEFT * 4).scale(0.8))
        bn_func = Text("BatchNorm(            )", font_size=60, color=TEAL).shift(LEFT * 2.5)
        self.play(FadeIn(bn_func))
        output_tensor = self.generate_tensor_cube(channel_colors=[TEAL, RED, GREEN])
        output_tensor.shift(4.5 * RIGHT).scale(0.4)
        arrow = Arrow(start=LEFT, end=RIGHT).shift(RIGHT * 2.2)
        self.play(FadeIn(arrow))
        self.play(FadeIn(output_tensor))

        if scale_back:
            input_tensor.scale(1 / 0.8)

        self.wait(1)
        # new_bn_func = Text("BatchNorm", font_size=60, color=TEAL).shift(LEFT * 2.5)
        self.play(
            # ReplacementTransform(bn_func, new_bn_func),
            FadeOut(bn_func),
            FadeOut(arrow),
            FadeOut(input_tensor),
            FadeOut(output_tensor),
        )
        self.wait()
        # return VGroup(new_bn_func, output_tensor)

    def switch_to_train(self):
        title = Text("2. Training process", font_size=90, color=BLUE)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

    def display_bn_train(self):
        self.switch_to_train()
        tensor1 = self.generate_random_3d_tensor(row=3, col=3, channel=3)
        batch1_text = Text("Batch = 1", color=TEAL).shift(DOWN * 2.5)
        self.play(FadeIn(tensor1))
        scale_factor = 0.75
        # self.play(tensor[-1].animate.shift(LEFT + DOWN))
        self.play(
            tensor1.animate.scale(scale_factor).shift(UP * 1.5 + LEFT * 2),
            FadeIn(batch1_text),
        )
        tensor2 = self.generate_random_3d_tensor(row=3, col=3, channel=3)
        batch2_text = Text("Batch = 2", color=TEAL).shift(DOWN * 2.5)
        self.play(FadeIn(tensor2))
        self.play(
            tensor2.animate.scale(scale_factor).shift(RIGHT * 2 + UP * 1.5),
            ReplacementTransform(batch1_text, batch2_text),
        )
        self.play(FadeOut(batch2_text))
        self.bn_stat(tensor1, tensor2)
        bn_attr, norm_tensor1, norm_tensor2 = self.normalize_tensor((tensor1, tensor2))
        self.play(
            norm_tensor1.animate.shift(DOWN * 1.5),
            norm_tensor2.animate.shift(DOWN * 1.5)
        )

        output_text = Text("output", font_size=80, color=BLUE).shift(LEFT * 1.5 + DOWN * 0.05)
        tensor_text = Text("tensor", font_size=80, color=BLUE).shift(RIGHT * 1.5)
        self.play(
            ReplacementTransform(norm_tensor1, output_text),
            ReplacementTransform(norm_tensor2, tensor_text),
            run_time=2,
        )
        self.wait()

        self.play(
            FadeOut(output_text),
            FadeOut(tensor_text),
        )
        self.bn_backward(bn_attr)

    def update_process(self):
        title = Text("3. Param update", font_size=90, color=BLUE)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

    def bn_backward(self, bn_attr):
        self.update_process()
        mean_and_var, weight_and_bias = bn_attr
        mean_and_var.shift(UP * 2.5)
        weight_and_bias.shift(UP * 2.5)
        self.play(
            FadeIn(mean_and_var),
            FadeIn(weight_and_bias),
        )
        title = Text("backward", font_size=80, color=BLUE).shift(UP * 1.5)
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))

        weight = self.generate_text_and_matix("weight: ", [[1.2, 1.9, 3.1]])
        weight.shift(RIGHT * 3)
        bias = self.generate_text_and_matix("bias: ", [[2.7, 2.2, 0.9]])
        bias.shift(DOWN + RIGHT * 3)
        new_w_and_b = VGroup(weight, bias)
        rect_w = SurroundingRectangle(weight_and_bias[0].matrix)
        rect_b = SurroundingRectangle(weight_and_bias[1].matrix)
        self.play(Create(rect_w), Create(rect_b), run_time=1)
        self.play(
            ReplacementTransform(weight_and_bias, new_w_and_b),
        )
        self.wait(3)
        self.play(FadeOut(rect_w), FadeOut(rect_b))
        self.play(FadeOut(new_w_and_b))

        # EMA update
        ema_text = Text("EMA update", font_size=80, color=BLUE).shift(UP * 1.5)
        self.play(
            Write(ema_text),
            mean_and_var.animate.shift(RIGHT * 3),
        )
        ema_des_text = Text("exponential moving average", color=TEAL, font_size=60).shift(DOWN * 2)
        ema_texts_copy = ema_text.copy()
        self.wait(2)
        self.play(ReplacementTransform(ema_texts_copy, ema_des_text))
        self.wait()
        self.play(FadeOut(ema_text, ema_des_text))
        self.play(mean_and_var.animate.shift(UP * 1))
        momentum_eq = MathTex(
            r"\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t",
            color=TEAL,
        )
        momentum_eq.shift(DOWN)
        self.play(Write(momentum_eq))
        x_hat_meaning = MathTex(
            r"\hat{x}: \text{running mean and variance}",
            color=BLUE,
        ).shift(DOWN * 2)
        x_t_meaning = MathTex(r"x_t: \text{input mean and variance}", color=BLUE).shift(DOWN * 3)
        self.play(
            Write(x_hat_meaning),
            Write(x_t_meaning),
        )
        self.wait(8)
        self.play(FadeOut(momentum_eq, x_hat_meaning, x_t_meaning))

        momentum_text = Text("momentum = 0. 1", font_size=50, color=BLUE, t2c={'0. 1': RED}).shift(DOWN * 1.5)
        self.play(Write(momentum_text))
        self.wait(4)
        mge_momentum_text = Text("momentum = 0. 9", font_size=50, color=BLUE, t2c={'0. 9': RED}).shift(DOWN * 1.5)
        in_mge_text = Text("in MegEngine", font_size=50, color=BLUE, t2c={" MegEngine": RED}).shift(DOWN * 2.5)
        merge_text = VGroup(mge_momentum_text, in_mge_text)
        self.play(
            ReplacementTransform(momentum_text, merge_text),
            run_time=2,
            # FadeIn(in_mge_text),
        )
        self.wait(1)
        self.play(FadeOut(merge_text))

        run_mean_val = [2.1, 1.7, 3.5]
        run_var_val = [4.1, 6.5, 2.9]
        running_mean = self.generate_text_and_matix("running mean: ", [run_mean_val])
        running_var = self.generate_text_and_matix("running var: ", [run_var_val])
        running_var.shift(DOWN)
        running_mean_and_var = VGroup(running_mean, running_var)
        running_mean_and_var.shift(DOWN * 1.5)
        self.play(FadeIn(running_mean_and_var), run_time=2)

        # momentum 0.1
        left_up_momentum = Text("0. 1 *", font_size=60, color=YELLOW).shift(LEFT * 4.5 + UP * 0.5)
        self.play(FadeIn(left_up_momentum),)
        rect_input_mean = SurroundingRectangle(mean_and_var[0].matrix)
        rect_input_var = SurroundingRectangle(mean_and_var[1].matrix)
        self.play(Create(rect_input_mean), Create(rect_input_var), run_time=1)

        input_mean_val = [0.44, 0.46, 0.43]
        input_var_val = [0.73, 0.8, 0.83]
        new_input_mean = self.generate_text_and_matix("input mean: ", [input_mean_val], num_decimal=2)
        new_input_var = self.generate_text_and_matix("input var: ", [input_var_val], num_decimal=2)
        new_input_mean.shift(UP)
        new_input_mean_and_var = VGroup(new_input_mean, new_input_var)
        self.play(ReplacementTransform(mean_and_var, new_input_mean_and_var))
        self.wait()
        self.play(
            FadeOut(rect_input_mean),
            FadeOut(rect_input_var),
            FadeOut(left_up_momentum),
        )

        # momentum 0.9
        left_down_momentum = Text("0. 9 *", font_size=60, color=YELLOW).shift(LEFT * 4.5 + DOWN * 2)
        self.play(FadeIn(left_down_momentum),)
        new_run_mean = self.generate_text_and_matix("running mean: ", [[x * 0.9 for x in run_mean_val]], num_decimal=2)
        new_run_var = self.generate_text_and_matix("running var: ", [[x * 0.9 for x in run_var_val]], num_decimal=2)
        new_run_var.shift(DOWN)
        rect_run_mean = SurroundingRectangle(running_mean.matrix)
        rect_run_var = SurroundingRectangle(running_var.matrix)
        self.play(Create(rect_run_mean), Create(rect_run_var), run_time=1)

        new_run_mean_and_var = VGroup(new_run_mean, new_run_var)
        new_run_mean_and_var.shift(DOWN * 1.5)
        self.play(ReplacementTransform(running_mean_and_var, new_run_mean_and_var))
        self.wait()
        self.play(
            FadeOut(rect_run_mean),
            FadeOut(rect_run_var),
            FadeOut(left_down_momentum),
        )

        # sum process
        sum_text = Text(" + ", font_size=60, color=YELLOW).shift(DOWN * 0.75)
        self.play(FadeIn(sum_text))
        self.wait()
        sum_mean_val = [x * 0.9 + y for x, y in zip(run_mean_val, input_mean_val)]
        sum_var_val = [x * 0.9 + y for x, y in zip(run_var_val, input_var_val)]
        sum_mean = self.generate_text_and_matix("running mean: ", [sum_mean_val], num_decimal=2)
        sum_var = self.generate_text_and_matix("running var: ", [sum_var_val], num_decimal=2)
        sum_var.shift(DOWN)
        sum_mean_and_var = VGroup(sum_mean, sum_var)
        sum_mean_and_var.shift(DOWN * 1.5)
        self.play(Create(rect_run_mean), Create(rect_run_var), run_time=1)
        self.play(ReplacementTransform(new_run_mean_and_var, sum_mean_and_var))

        self.play(
            FadeOut(rect_run_mean),
            FadeOut(rect_run_var),
            FadeOut(sum_text),
        )
        self.wait()
        self.play(FadeOut(new_input_mean_and_var))
        self.play(sum_mean_and_var.animate.shift(LEFT * 3 + UP * 1.5), FadeIn(new_w_and_b),)
        self.wait()
        params = VGroup(sum_mean_and_var, new_w_and_b)
        param_text = Text("Parameters of BatchNorm", font_size=60, color=BLUE)
        self.play(ReplacementTransform(params, param_text), run_time=2)
        self.wait(2)
        self.play(FadeOut(param_text))

    def normalize_tensor(self, tensors, bn_weight=[[1.0, 2.0, 3.0]], bn_bias=[[3.0, 2.0, 1.0]], training=True):
        str_prefix = "input" if training else "running"
        if training:
            bn_eq = MathTex(
                r"y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + {\epsilon} }} * \gamma + \beta",
                color=TEAL,
            ).shift(DOWN * 1.25)
            self.play(Write(bn_eq), run_time=3)
            bn_eq_mean = MathTex(
                r"\gamma: \text{weight of BatchNorm   } \beta: \text{bias of BatchNorm}",
                color=BLUE,
            ).shift(DOWN * 3)
            self.wait()
            self.play(Write(bn_eq_mean))
            self.wait(14)
            self.play(FadeOut(bn_eq_mean))
            eps_text = MathTex(
                r"\epsilon: \text{avoid NaN for numerical stability}",
                color=RED,
            ).shift(DOWN * 3)
            self.play(Write(eps_text))
            self.wait(7)
            self.play(
                bn_eq.animate.scale(0.75).shift(LEFT * 4.5),
                FadeOut(eps_text),
            )

        weight = self.generate_text_and_matix("weight: ", bn_weight)
        weight.matrix.value = bn_weight
        bias = self.generate_text_and_matix("bias: ", bn_bias)
        bias.matrix.value = bn_bias

        weight.shift(DOWN * 2.5)
        bias.shift(DOWN * 3.5)
        weight_and_bias = VGroup(weight, bias)
        weight_and_bias.shift(RIGHT * 3)

        def get_mean(tensors):
            mean = []
            for m1, m2 in zip(*tensors):
                m_value = np.concatenate((m1.value, m2.value)).flatten()
                mean.append(m_value.mean())
            return [mean]

        def get_var(tensors):
            var = []
            for m1, m2 in zip(*tensors):
                m_value = np.concatenate((m1.value, m2.value)).flatten()
                var.append(m_value.var())
            return [var]

        if training:
            mean_value = get_mean(tensors)
        else:
            mean_value = [[1.4, 3.3, 2.2]]
        mean = self.generate_text_and_matix(str_prefix + " mean: ", mean_value)
        mean.matrix.value = mean_value 

        if training:
            var_value = get_var(tensors)
        else:
            var_value = [[4.2, 5.4, 3.8]]

        var = self.generate_text_and_matix(str_prefix + " var: ", var_value)
        var.matrix.value = var_value

        mean.shift(DOWN * 2.5)
        var.shift(DOWN * 3.5)
        mean_and_var = VGroup(mean, var)
        mean_and_var.shift(LEFT * 3)

        self.play(
            FadeIn(mean_and_var),
            FadeIn(weight_and_bias),
        )
        new_tensor_list1, new_tensor_list2 = [], []
        for channel, (m1, m2) in enumerate(zip(*tensors)):
            matrix_shift = 1.8 + (2 - channel) * 0.5

            m_value = np.concatenate((m1.value, m2.value)).flatten()
            if training:
                mean_val, var_val = m_value.mean(), m_value.var()
            else:
                mean_val, var_val = mean.matrix.value[0][channel], var.matrix.value[0][channel]

            matrix_value = np.round((m1.value - mean_val) / np.sqrt(var_val) * weight.matrix.value[0][-1-channel] + bias.matrix.value[0][-1-channel], 1)
            new_m1 = Matrix(matrix_value, fill_opacity=1, include_background_rectangle=True)
            matrix_value = np.round((m2.value - mean_val) / np.sqrt(var_val) * weight.matrix.value[0][-1-channel] + bias.matrix.value[0][-1-channel], 1)
            new_m2 = Matrix(matrix_value, fill_opacity=1, include_background_rectangle=True)
            new_m2.value = matrix_value

            shift_vector = DOWN * matrix_shift
            if training:
                shift_vector += RIGHT * (channel * 0.5 + 1)
            self.play(
                m1.animate.shift(shift_vector),
                m2.animate.shift(shift_vector),
            )
            self.wait()

            if channel == 0:
                rect_mean = SurroundingRectangle(mean_and_var[0].matrix.get_rows()[0][-1])
                rect_var = SurroundingRectangle(mean_and_var[1].matrix.get_rows()[0][-1])
                rect_w = SurroundingRectangle(weight.matrix.get_rows()[0][-1])
                rect_b = SurroundingRectangle(bias.matrix.get_rows()[0][-1])
                self.play(
                    Create(rect_mean), Create(rect_var),
                    Create(rect_w), Create(rect_b),
                )
            else:
                new_rect_mean = SurroundingRectangle(mean_and_var[0].matrix.get_rows()[0][-1-channel])
                new_rect_var = SurroundingRectangle(mean_and_var[1].matrix.get_rows()[0][-1-channel])
                new_rect_w = SurroundingRectangle(weight.matrix.get_rows()[0][-1-channel])
                new_rect_b = SurroundingRectangle(bias.matrix.get_rows()[0][-1-channel])
                self.play(
                    ReplacementTransform(rect_mean, new_rect_mean),
                    ReplacementTransform(rect_var, new_rect_var),
                    ReplacementTransform(rect_w, new_rect_w),
                    ReplacementTransform(rect_b, new_rect_b),
                )
                rect_mean, rect_var = new_rect_mean, new_rect_var
                rect_w, rect_b = new_rect_w, new_rect_b

            new_m1.scale(0.75).shift(m1.get_center())
            new_m2.scale(0.75).shift(m2.get_center())
            self.play(
                ReplacementTransform(m1, new_m1),
                ReplacementTransform(m2, new_m2),
                run_second=2
            )
            new_tensor_list1.append(new_m1)
            new_tensor_list2.append(new_m2)
            self.wait()
            self.play(
                new_m1.animate.shift(-shift_vector),
                new_m2.animate.shift(-shift_vector),
            )
        self.play(
            FadeOut(rect_mean), FadeOut(rect_var),
            FadeOut(rect_w), FadeOut(rect_b),
        )
        if training:
            self.play(
                FadeOut(bn_eq),
                FadeOut(mean_and_var),
                FadeOut(weight_and_bias),
            )
        else:
            self.play(
                FadeOut(mean_and_var),
                FadeOut(weight_and_bias),
            )

        return (mean_and_var, weight_and_bias), VGroup(*new_tensor_list1), VGroup(*new_tensor_list2)
    
    def generate_text_and_matix(self, text, matrix_value=None, num_decimal=1, t2c={}):
        if matrix_value is None:
            matrix_value = [[0.0, 0.0, 0.0]]

        t2c.update({"running": RED, "input": RED,})
        text_obj = Text(text, t2c=t2c).shift(DOWN * 12) 
        matrix_obj = Matrix(
            matrix_value, fill_opacity=1,
            element_to_mobject=functools.partial(DecimalNumber, num_decimal_places=num_decimal),
        )
        text_obj.next_to(matrix_obj, LEFT)
        group = VGroup(text_obj, matrix_obj).arrange(center=True, buff=1).scale(0.7)
        group.text = text_obj
        group.matrix = matrix_obj
        return group

    def bn_stat(self, tensor1, tensor2):
        mean = self.generate_text_and_matix(text="input mean: ")
        var = self.generate_text_and_matix(text="input var: ")

        # TODO comment next 3 lines
        # mean_and_var = VGroup(mean, var)
        # return mean_and_var
        expect_eq = MathTex(
            r"\mathrm{E}[X] = \frac{x_1 + x_2 + ... + x_n}{n}",
            color=TEAL,
        ).shift(DOWN * 1.5)

        var_eq = MathTex(
            r"\mathrm{Var}[X] = \mathrm{E}[X^2] - (\mathrm{E}[X])^2",
            color=TEAL,
        ).shift(DOWN * 2.5)
        var_def_eq = MathTex(
            r"\mathrm{Var}[X] = \frac{\displaystyle\sum_{i=1}^{n}(x_i - \mathrm{E}[X])^2} {n}",
            color=TEAL,
        ).shift(DOWN * 1.5)
        self.wait(3)
        self.play(
            Write(expect_eq),
            Write(var_eq),
        )

        # next few line to wait(20)
        self.wait(4)
        round_var_eq = SurroundingRectangle(var_eq)
        self.play(Create(round_var_eq))
        self.wait()
        self.play(FadeOut(round_var_eq))
        var_eq_copy = var_eq.copy()
        self.play(ReplacementTransform(var_eq, var_def_eq), FadeOut(expect_eq))
        self.wait(2)
        self.play(ReplacementTransform(var_def_eq, var_eq_copy), FadeIn(expect_eq))

        self.wait(8)
        self.play(FadeOut(var_eq_copy), FadeOut(expect_eq))

        mean.shift(DOWN * 2.5)
        var.shift(DOWN * 3.5)

        for channel in range(3):
            m1, m2 = tensor1[-1 - channel], tensor2[-1 - channel]
            matrix_shift = 1.8 + channel * 0.5

            channel_info = Text(f"channel index: {channel + 1}", color=BLUE).scale(0.8).shift(DOWN * 1.2)
            self.play(FadeIn(channel_info))
            self.wait()
            self.play(FadeOut(channel_info))

            # self.play()

            mean_tracker = ValueTracker(0)
            var_tracker = ValueTracker(0)
            mean.matrix[0][channel].add_updater(lambda d: d.set_value(mean_tracker.get_value()))
            var.matrix[0][channel].add_updater(lambda d: d.set_value(var_tracker.get_value()))

            if channel != 0:
                mean.matrix.update_value.suspend_updating()
                var.matrix.update_value.suspend_updating()

            mean.matrix.update_value = mean.matrix[0][channel]
            var.matrix.update_value = var.matrix[0][channel]

            self.play(
                m1.animate.shift(DOWN * matrix_shift),
                m2.animate.shift(DOWN * matrix_shift),
            )

            if channel == 0:
                self.play(FadeIn(mean), FadeIn(var))

            value_list = list()
            for row in range(3):
                for col in range(3):
                    if channel == 0:
                        if row == 0 and col == 0:
                            rect_m1 = SurroundingRectangle(m1.get_rows()[0][0])
                            rect_m2 = SurroundingRectangle(m2.get_rows()[0][0])
                            rect_mean = SurroundingRectangle(mean.matrix.get_rows()[0][channel])
                            rect_var = SurroundingRectangle(var.matrix.get_rows()[0][channel])
                            self.play(
                                Create(rect_m1), Create(rect_m2),
                                Create(rect_mean), Create(rect_var),
                            )
                            value_list.extend([m1.value[0][0], m2.value[0][0]])
                            self.play(
                                mean_tracker.animate.set_value(sum(value_list) / len(value_list)),
                                var_tracker.animate.set_value(np.var(value_list)),
                            )
                            continue

                        new_rect_m1 = SurroundingRectangle(m1.get_rows()[row][col])
                        new_rect_m2 = SurroundingRectangle(m2.get_rows()[row][col])
                        value_list.extend([m1.value[row][col], m2.value[row][col]])
                        self.play(
                            ReplacementTransform(rect_m1, new_rect_m1),
                            ReplacementTransform(rect_m2, new_rect_m2),
                        )
                        self.play(
                            mean_tracker.animate.set_value(sum(value_list) / len(value_list)),
                            var_tracker.animate.set_value(np.var(value_list)),
                        )
                        rect_m1, rect_m2 = new_rect_m1, new_rect_m2
                    else:
                        if row == 0 and col == 0:
                            rect_mean = SurroundingRectangle(mean.matrix.get_rows()[0][channel])
                            rect_var = SurroundingRectangle(var.matrix.get_rows()[0][channel])
                            self.play(Create(rect_mean), Create(rect_var))
                            value_list = [_ for x in m1.value for _ in x] + [_ for x in m2.value for _ in x]
                            self.play(
                                mean_tracker.animate.set_value(sum(value_list) / len(value_list)),
                                var_tracker.animate.set_value(np.var(value_list)),
                            )
                        else:
                            continue
            
            if channel == 0:
                self.play(FadeOut(rect_m1, rect_m2, rect_mean, rect_var))
            else:
                self.play(FadeOut(rect_mean, rect_var))

            self.play(
                m1.animate.shift(UP * matrix_shift),
                m2.animate.shift(UP * matrix_shift),
            )

        mean_and_var = VGroup(mean, var)
        self.play(FadeOut(mean_and_var))

    def switch_to_eval(self):
        title = Text("4. Inference process", font_size=90, color=BLUE)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

    def display_bn_eval(self):
        self.switch_to_eval()
        input_text = Text("input", font_size=80, color=BLUE).shift(LEFT * 1.5).shift(DOWN * 0.05)
        tensor_text = Text("tensor", font_size=80, color=BLUE).shift(RIGHT * 1.5)
        self.play(Write(input_text), Write(tensor_text), run_second=2)
        tensor1 = self.generate_random_3d_tensor(row=3, col=3, channel=3)
        scale_factor = 0.75
        tensor1.scale(scale_factor).shift(UP * 1.5 + LEFT * 2)
        tensor2 = self.generate_random_3d_tensor(row=3, col=3, channel=3)
        tensor2.scale(scale_factor).shift(RIGHT * 2 + UP * 1.5)
        self.play(
            ReplacementTransform(input_text, tensor1),
            ReplacementTransform(tensor_text, tensor2),
        )
        _, norm_tensor1, norm_tensor2 = self.normalize_tensor(
            (tensor1, tensor2),
            bn_weight=[[1.5, 0.7, 2.8]],
            bn_bias=[[0.4, 1.1, 0.6]],
            training=False,
        )
        output_text = Text("output", font_size=80, color=TEAL).shift(LEFT * 1.5 + DOWN * 0.05)
        tensor_text = Text("tensor", font_size=80, color=TEAL).shift(RIGHT * 1.5)
        self.play(
            ReplacementTransform(norm_tensor1, output_text),
            ReplacementTransform(norm_tensor2, tensor_text),
        )
        self.wait()
        self.play(FadeOut(output_text), FadeOut(tensor_text))

    def display_thanks(self):
        reference_text = Text(
            "Reference:",
            font_size=40,
            color=BLUE,
        ).shift(UP * 2 + LEFT * 5.5)
        bn_paper = Text(
            "[1]: Ioffe, Sergey, and Christian Szegedy. \n\"Batch normalization: Accelerating deep network \n"\
            "training by reducing internal covariate shift.\"\nInternational conference on machine learning. PMLR, 2015.",
            font_size=30,
            color=BLUE,
        ).shift(LEFT * 1.6)
        torch_doc = Text(
            "[2]: PyTorch 1.10.0 Documentation, \n"\
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html",
            font_size=30,
            color=BLUE,
        )
        mge_title = Text(
            "[3]: MegEngine 1.6 Documentation,\n",
            font_size=30,
            color=BLUE,
        )
        mge_url = Text(
            "https://megengine.org.cn/doc/stable/zh/reference/api/megengine.module.BatchNorm2d.html",
            font_size=27,
            color=BLUE,
        )
        mge_doc = VGroup(mge_title, mge_url).arrange(DOWN, center=False, aligned_edge=LEFT)
        VGroup(bn_paper, torch_doc, mge_doc).arrange(DOWN, center=False, aligned_edge=LEFT)
        self.play(Write(reference_text))
        self.play(
            Write(bn_paper),
            Write(torch_doc),
            Write(mge_doc),
        )
        self.wait(2.5)
        self.play(FadeOut(reference_text, bn_paper, torch_doc, mge_doc))

        thanks = Text("Thanks for watching", font_size=90, color=BLUE)
        self.play(Write(thanks))
        self.wait(1)
        self.play(FadeOut(thanks))

    def display_timeline(self):
        kwargs = {"font_size": 35, "color": TEAL}
        timeline = Text("Timeline", font_size=50, color=BLUE).shift(UP * 2)
        concept = Text("1. Concept", **kwargs)
        train = Text("2. Training process", **kwargs)
        update = Text("3. Param update", **kwargs)
        inference = Text("4. Inference process", **kwargs)
        content = VGroup(concept, train, update, inference).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        content.shift(DOWN)
        self.play(Write(timeline))
        self.play(FadeIn(content))
        self.wait(1)
        self.play(FadeOut(timeline))
        return content

    def display_bilibili(self):
        thumb_up = ImageMobject("thumb.png").scale(0.5).shift(LEFT * 3)
        coin = ImageMobject("coin.png").scale(0.5)
        star = ImageMobject("star.png").scale(0.5).shift(RIGHT * 3)
        self.play(FadeIn(thumb_up), FadeIn(coin), FadeIn(star))
        kwargs = {
            "flash_radius": 1, "num_lines": 30, "color": BLUE,
            "run_time": 2,
            "line_length": 0.5,
        }
        self.play(
            Flash(thumb_up, **kwargs),
            Flash(coin, **kwargs),
            Flash(star, **kwargs),
        )
        self.wait(1)
        text_kwargs = {"font_size": 30, "color": BLUE}
        thumb_text = Text("模型收敛", **text_kwargs)
        coin_text = Text("算法涨点", **text_kwargs)
        star_text = Text("效果逆天", **text_kwargs)
        thumb_text.next_to(thumb_up, UP)
        coin_text.next_to(coin, UP)
        star_text.next_to(star, UP)
        self.play(
            Write(thumb_text),
            Write(coin_text),
            Write(star_text),
        )
        self.wait(2)

    def construct(self):
        title = self.title()
        content = self.display_timeline()

        tensor_cube = self.batch_concept(content)
        self.display_bn_func(tensor_cube)

        self.display_bn_train()
        self.display_bn_eval()

        self.play(FadeOut(title))
        self.display_thanks()
        self.display_bilibili()
