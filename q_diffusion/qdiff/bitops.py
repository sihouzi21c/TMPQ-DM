def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res
def quantmodule_hook(layer,input,output):
    ins = input[0].size()
    outs = output.size()
    #idx=self.use_quantizer_idx
    weights=F.softmax(layer.weight_quantizer_alpha,dim=-1)
    activations=F.softmax(layer.act_quantizer_alpha,dim=-1)
    bitw=sum([weights[i]*layer.weight_quantizer_list[i].n_bits for i in range(len(layer.weight_quantizer_list))])
    bita=sum([activations[i]*layer.act_quantizer_list[i].n_bits for i in range(len(layer.act_quantizer_list))])
    #bitw=self.weight_quantizer_list[idx].n_bits
    #bita=self.act_quantizer_list[idx].n_bits
    if layer.idx==0: #Conv2d
        # thop method
        input_size = list(input[0].shape)
        output_size = list(output.shape)
        kernel_size=layer.kernel_size
        groups=layer.groups
        in_c=input_size[1]
        g=groups
        n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
        n_bitops = n_macs * bitw * bita * 1e-9
        list_conv2d.append(n_bitops)
                
    elif layer.idx==1: #Conv1d: not used
        print('i have')
        input_size = list(input[0].shape)
        output_size = list(output.shape)
        kernel_size=layer.kernel_size
        groups=layer.groups
        in_c=input_size[1]
        g=groups
        n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
        n_bitops=n_macs*bitw*bita*1e-9
        list_conv1d.append(n_bitops)
    elif layer.idx==2:  #Linear:
                #thop
        total_mul = layer.in_features
        num_elements = output.numel()
        def calculate_linear(total_mul, num_elements):
            return int(total_mul * num_elements)
        n_macs=calculate_linear(total_mul, num_elements)
        n_bitops = n_macs * bitw * bita * 1e-9
        list_linear.append(n_bitops)
                
def basequantblock_hook(block,input,output):#not used
    b, c, *spatial = output[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_macs = 2 * b * (num_spatial ** 2) * c
    idx=block.use_quantizer_idx
            
    bita=block.act_quantizer_q_list[idx].n_bits
    matmul_ops=matmul_macs*bita*bita*1e-9
    list_QKV.append(matmul_ops)
            
def quantattnblock_hook(block,input,output):
            # b,c, *spatial = output[0].shape #256,16,16  疑惑？why都是这样
            # num_spatial = int(np.prod(spatial))
            # matmul_macs = 2 * b * (num_spatial ** 2) *c 
    b, *spatial = output[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_macs = b * b * num_spatial
            #idx=self.use_quantizer_idx
            #qk 一次，vw一次
    act_q_weights=F.softmax(block.act_q_weights)
    act_k_weights=F.softmax(block.act_k_weights)
    act_v_weights=F.softmax(block.act_v_weights)
    act_w_weights=F.softmax(block.act_w_weights)
    bitq=sum([act_q_weights[i]*block.act_quantizer_q_list[i].n_bits for i in range(len(act_q_weights))])
    bitk=sum([act_k_weights[i]*block.act_quantizer_k_list[i].n_bits for i in range(len(act_k_weights))])
    bitv=sum([act_v_weights[i]*block.act_quantizer_v_list[i].n_bits for i in range(len(act_v_weights))])
    bitw=sum([act_w_weights[i]*block.act_quantizer_w_list[i].n_bits for i in range(len(act_w_weights))])
    matmul_ops=matmul_macs*(bitq*bitk+bitv*bitw)*1e-9
            ##matmul_macs = 2 * b * b * num_spatial
            ##idx=self.use_quantizer_idx
            ##bita=self.act_quantizer_q_list[idx].n_bits
            ##matmul_ops=matmul_macs*bita*bita*1e-9
    list_QKV.append(matmul_ops)

handler_list=[]
        #for i in range(len(use_bitwidth)):
            #use_bitwidth[i]
            #cnt=0
for m in self.model.modules():
        #if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
     if isinstance(m,(QuantModule_darts,BaseQuantBlock_darts)):
                #m.set_quant_idx(use_bitwidth[i][cnt])
                #cnt+=1
        if isinstance(m,QuantModule_darts):
            handler=m.register_forward_hook(quantmodule_hook)
            handler_list.append(handler)
        elif isinstance(m,QuantAttnBlock_darts):
            handler=m.register_forward_hook(quantattnblock_hook)
            handler_list.append(handler)