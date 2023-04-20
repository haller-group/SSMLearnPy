def encode_geometry(
    encoder, 
    x
    ):
    x_reduced = []
    for i_elem in range(len(x)):
        x_reduced.append(encoder(x[i_elem]))
    return x_reduced

def decode_geometry(
    decoder, 
    x_reduced
    ):
    x = []
    for i_elem in range(len(x_reduced)):
        x.append(decoder(x_reduced[i_elem].T).T)
    return x    