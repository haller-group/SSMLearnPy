def function_to_list(
    fun, 
    x_in
    ):
    x_out = []
    for i_elem in range(len(x_in)):
        x_out.append(fun(x_in[i_elem]))
    return x_out


