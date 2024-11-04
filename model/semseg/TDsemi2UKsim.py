
import torch
from torch import nn
import torch.nn.functional as F
from model.semseg.cstmformer import TemporalTransformer


class Labelsim(nn.Module):
    def __init__(self, cfg, fea_dim):
        super(Labelsim, self).__init__()
        self.nclass = cfg['nclass']
        eps = 1e-8
        centerf_init = eps * (2 * torch.randint(0, 2, (cfg['nclass'], fea_dim)) - 1)
        self.register_parameter('centerf', nn.Parameter(centerf_init))
    
    def semilabel(self, seglabel, x):
        sim_all_loss = torch.tensor(0).cuda().float()
        index = torch.nonzero(seglabel == 0)
        feature_i = torch.squeeze(x[index])
        sim = torch.mm(F.normalize(feature_i, dim=-1), F.normalize(self.centerf, dim = -1).t())[:,1:]
        sim_max, pro_label = torch.max(sim, dim=1)
        mask_is0 = ~(sim_max > 0.9)
        is0 = index[mask_is0]
        not0 = index[~mask_is0]
        if not is0.numel() == 0:
            sim_0 = 1 - F.cosine_similarity(F.normalize(torch.squeeze(x[is0]), dim=-1), F.normalize(torch.unsqueeze(self.centerf[0], dim = 0), dim=-1), dim=-1)
            sim_all_loss += torch.sum(sim_0)
        if not not0.numel() == 0:
            seglabel[not0] = torch.unsqueeze((pro_label + 1)[~mask_is0], dim = 1)
        return seglabel, sim_all_loss

    def forward(self, xA, xB, seglabelA, seglabelB, begin_train = None):
        if seglabelA == None:
            return self.centerf
        else:
            sim_all_loss = torch.tensor(0).cuda().float()

            for i in range(1, self.nclass):
                index_i_A = torch.nonzero(seglabelA == i)
                feature_i_A = torch.squeeze(xA[index_i_A])
                sim_A = 1 - F.cosine_similarity(F.normalize(feature_i_A, dim=-1), F.normalize(torch.unsqueeze(self.centerf[i], dim = 0), dim=-1), dim=-1)
                sim_all_loss += torch.sum(sim_A)

                index_i_B = torch.nonzero(seglabelB == i)
                feature_i_B = torch.squeeze(xB[index_i_B])
                sim_B = 1 - F.cosine_similarity(F.normalize(feature_i_B, dim=-1), F.normalize(torch.unsqueeze(self.centerf[i], dim = 0), dim=-1), dim=-1)
                sim_all_loss += torch.sum(sim_B)

            if begin_train == True:
                seglabelA, sim_all_loss_A = self.semilabel(seglabelA, xA)
                seglabelB, sim_all_loss_B = self.semilabel(seglabelB, xB)

                sim_all_loss += sim_all_loss_A
                sim_all_loss += sim_all_loss_B

            else: 
                seglabelA, seglabelB = None, None

            return sim_all_loss, self.centerf, seglabelA, seglabelB


class DGConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, sort=False):
        super(DGConv3d, self).__init__()
        self.register_buffer('D', torch.eye(out_channels, in_channels))
        self.register_buffer('I', torch.ones(out_channels, in_channels))
        eps = 1e-8
        
        gate_init = eps * (2 * torch.randint(0, 2, (out_channels, in_channels)) - 1)
        #gate_init = 2 * torch.rand(4, 6) - 1  
        self.register_parameter('gate', nn.Parameter(gate_init))
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sort = sort

        self.C = out_channels #sum(range(1, band_extra))

        relax_denom = 0

        self.i = torch.eye(self.C, self.C).cuda()
        self.reversal_i = torch.ones(self.C, self.C).triu(diagonal=1).cuda()
        self.num_off_diagonal = torch.sum(self.reversal_i)
        if relax_denom == 0:
            #print("Note relax_denom == 0!")
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom

        #self.bn  = nn.BatchNorm3d(out_channels)

    def get_covariance_matrix(self, x):
        eps = 1e-5
        B, C, T, H, W = x.shape  # i-th feature size (B X C X H X W)
        THW = T * H * W
        eye = torch.eye(C).cuda()
        x = x.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        x_cor = torch.bmm(x, x.transpose(1, 2)).div(THW - 1) + (eps * eye)  # C X C / HW

        return x_cor, B
     
    def instance_whitening_loss(self, x, mask_matrix, margin, num_remove_cov):
        x_cor, B = self.get_covariance_matrix(x)
        x_cor_masked = x_cor * mask_matrix

        off_diag_sum = torch.sum(torch.abs(x_cor_masked), dim=(1, 2), keepdim=True) - margin  # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0)  # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss

    def forward(self, x):
        setattr(self.gate, 'org', self.gate.data.clone())
        #self.gate.data = nn.Softmax()
        self.gate.data = ((self.gate.org - 0).sign() + 1) / 2.
        #U_regularizer =  2 ** (self.K  + torch.sum(self.gate))
        gate = torch.stack((1 - self.gate, self.gate))
        self.gate.data = self.gate.org
        U = gate[0, :] * self.D + gate[1, :] * self.I
        #print(U)
        masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1, 1)
        #x = F.conv3d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
        x_size = x.size()
        masked_weight = F.normalize(torch.squeeze(masked_weight), p = 2)
#

        x = x.contiguous().view(x_size[0], x_size[1], -1).transpose(2, 1).reshape(-1,x_size[1])
        x = torch.matmul(x, masked_weight.transpose(1, 0))


        #x = x.view(x_size[0], x_size[1], -1).transpose(2, 1).reshape(-1,x_size[1])

        x_normalize = F.normalize(x, dim = 0, p=2.0, eps=1e-6)

        x_normalize = x_normalize.view(x_size[0], -1 , x_size[1]).transpose(2, 1)
        
        x_normalize = x_normalize.contiguous().view(x_size[0], -1, x_size[2], x_size[3], x_size[4])
        
        loss = self.instance_whitening_loss(x_normalize, self.reversal_i, self.margin, self.num_off_diagonal)

        return x_normalize, loss #, U_regularizer

class bbs(nn.Module):
     def __init__(self, band_extra, relax_denom = 0):
        super(bbs, self).__init__()

        self.C = sum(range(1, band_extra))

        self.i = torch.eye(self.C, self.C).cuda()
        self.reversal_i = torch.ones(self.C, self.C).triu(diagonal=1).cuda()
        self.num_off_diagonal = torch.sum(self.reversal_i)
        if relax_denom == 0:
            #print("Note relax_denom == 0!")
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom

        self.extra = band_extra


     def bilinear_pooling(self, x, y):
         x_size = x.size()
         y_size = y.size()
         
         out = torch.mul(x.unsqueeze(2), (1/(y+1e-6)).unsqueeze(1))

         indic1 = torch.triu_indices(out.shape[1], out.shape[2], offset=1)[0]
         indic2 = torch.triu_indices(out.shape[1], out.shape[2], offset=1)[1]

         out_l = out[:, indic1, indic2, :, :, :]
         #out2 = out[:, indic2, indic1, :, :, :]
         #out_l = torch.cat([out1, out2], dim = 1)

         #out_normalize = F.normalize(out, dim = 1, p=2.0, eps=1e-6)#

         out_l = out_l.view(x_size[0], self.C, -1).transpose(1,2).reshape(-1, self.C)

         out_normalize = F.normalize(out_l, dim = 0, p=2.0, eps=1e-6)

         out_normalize = out_normalize.view(x_size[0], -1, self.C).transpose(1, 2).view(x_size[0], self.C, x_size[2], x_size[3], x_size[4])

     
         return out_normalize   #[N,C,F*F]
     
     
     def get_covariance_matrix(self, x):
        eps = 1e-5
        B, C, T, H, W = x.shape  # i-th feature size (B X C X H X W)
        THW = T * H * W
        eye = torch.eye(C).cuda()
        x = x.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        x_cor = torch.bmm(x, x.transpose(1, 2)).div(THW - 1) + (eps * eye)  # C X C / HW

        return x_cor, B
     
     def instance_whitening_loss(self, x, mask_matrix, margin, num_remove_cov):
        x_cor, B = self.get_covariance_matrix(x)
        x_cor_masked = x_cor * mask_matrix

        off_diag_sum = torch.sum(torch.abs(x_cor_masked), dim=(1, 2), keepdim=True) - margin  # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0)  # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss
     
     def forward(self, x):
         
         out = self.bilinear_pooling(x, x)

         loss = self.instance_whitening_loss(out, self.reversal_i, self.margin, self.num_off_diagonal)

         return out, loss


class TDsemi2UKsim(nn.Module):
    def __init__(self, cfg):
        super(TDsemi2UKsim, self).__init__()

      
        channel_size = (32, 64, 256)
        kernel_size = (5, 3, 5, 3)


        self.convb1 = DGConv3d(6, cfg['extra_band'], 1)
        self.convb2 = DGConv3d(6, cfg['extra_band'], 1)
        
        
        self.bbs1 = bbs(cfg['extra_band'], 0)
        self.bbs2 = bbs(cfg['extra_band'], 0)

        self.DGCnor = nn.Sequential(nn.BatchNorm3d(3))
        

        self.convx11 = nn.Sequential(
                nn.Conv3d(in_channels=6 + sum(range(1, cfg['extra_band'])),#+ cfg['extra_band'] ,#, #
                        out_channels=channel_size[0],
                        kernel_size=(kernel_size[0], kernel_size[1], kernel_size[1]),
                        padding = 1),
                nn.ReLU(),
                nn.BatchNorm3d(channel_size[0]),
            )
        
        self.convx12 = nn.Sequential(
                nn.Conv3d(in_channels=channel_size[0],
                        out_channels=channel_size[1],
                        kernel_size=(kernel_size[0], kernel_size[1], kernel_size[1]),
                        ),
                nn.ReLU(),
                nn.BatchNorm3d(channel_size[1]),
            )
        
        self.convx21 = nn.Sequential(
                nn.Conv3d(in_channels=6 + sum(range(1, cfg['extra_band'])), #+ cfg['extra_band'],#,#, #
                        out_channels=channel_size[0],
                        kernel_size=(kernel_size[0], kernel_size[1], kernel_size[1]),
                        padding = 1),
                nn.ReLU(),
                nn.BatchNorm3d(channel_size[0]),
            )
        
        self.convx22 = nn.Sequential(
                nn.Conv3d(in_channels=channel_size[0],
                        out_channels=channel_size[1],
                        kernel_size=(kernel_size[0], kernel_size[1], kernel_size[1]),
                       ),
                nn.ReLU(),
                nn.BatchNorm3d(channel_size[1]),
            )
        
        self.convseg = nn.Sequential(
                nn.Conv3d(in_channels=channel_size[1],
                        out_channels=channel_size[2],
                        kernel_size=(kernel_size[0], kernel_size[3], kernel_size[3])),
                nn.ReLU(),
                nn.BatchNorm3d(channel_size[2]),
                )

        self.convbn = nn.Sequential(
                nn.Conv3d(in_channels=channel_size[1]*2,
                        out_channels=channel_size[2]*2,
                        kernel_size=(kernel_size[0], kernel_size[3], kernel_size[3])),
                nn.ReLU(),
                nn.BatchNorm3d(channel_size[2]*2),
                )
        
        dropout=0.1
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.linears = nn.Linear(in_features=512,#channel_size[1]*2
                                out_features=128)
        
        self.linearc = nn.Linear(in_features=1024,#channel_size[1]*2
                                out_features=256)


        self.classifier_bn = nn.Linear(256, 7)
        self.classifier_ss = nn.Linear(128, cfg['nclass'])

        self.labelsim = Labelsim(cfg, 128)

        self.cstm11 = TemporalTransformer(width=32, layers=1, heads=8)
        self.cstm21 = TemporalTransformer(width=32, layers=1, heads=8)
        relax_denom = 0
        self.reversal_i = torch.ones(6 + sum(range(1, cfg['extra_band'])), 6 + sum(range(1, cfg['extra_band']))).triu(diagonal=1).cuda()
        self.num_off_diagonal = torch.sum(self.reversal_i)
        if relax_denom == 0:
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom
    
    def get_covariance_matrix(self, x):
        eps = 1e-5
        B, C, T, H, W = x.shape  # i-th feature size (B X C X H X W)
        THW = T * H * W
        eye = torch.eye(C).cuda()
        x = x.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        x_cor = torch.bmm(x, x.transpose(1, 2)).div(THW - 1) + (eps * eye)  # C X C / HW

        return x_cor, B
     
    def instance_whitening_loss(self, x, mask_matrix, margin, num_remove_cov):
        x_cor, B = self.get_covariance_matrix(x)
        x_cor_masked = x_cor * mask_matrix

        off_diag_sum = torch.sum(torch.abs(x_cor_masked), dim=(1, 2), keepdim=True) - margin  # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0)  # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss
     
    def forward(self, x1, x2, labels_A, labels_B, begin_train = None):
        bs = x1.shape[0]
        t = x1.shape[1]
        f = x1.shape[2]
        h, w = x1.shape[-2:]

        x1 = x1.transpose(2, 1)
        x2 = x2.transpose(2, 1)

        x1_band, loss1 = self.convb1(x1)
        x2_band, loss2 = self.convb1(x2)

        #loss = (loss1 + loss2)/2
   
        x1_band_bp, loss1 = self.bbs1(x1_band)
        x2_band_bp, loss2 = self.bbs1(x2_band)
        
        loss = (loss1 + loss2)/2
        
        #x1_band_bp1 = torch.unsqueeze((x1[:,3,:,:,:] - x1[:,2,:,:,:])/ (x1[:,3,:,:,:] + x1[:,2,:,:,:] + 1e-6), dim = 1)
        #x1_band_bp2 = torch.unsqueeze((x1[:,1,:,:,:] - x1[:,3,:,:,:])/ (x1[:,1,:,:,:] + x1[:,3,:,:,:] + 1e-6), dim = 1)
        #x1_band_bp = torch.cat((x1_band_bp1, x1_band_bp2), dim = 1)
        ###
        #x2_band_bp1 = torch.unsqueeze((x2[:,3,:,:,:] - x2[:,2,:,:,:])/ (x2[:,3,:,:,:] + x2[:,2,:,:,:] + 1e-6), dim = 1)
        #x2_band_bp2 = torch.unsqueeze((x2[:,1,:,:,:] - x2[:,3,:,:,:])/ (x2[:,1,:,:,:] + x2[:,3,:,:,:] + 1e-6), dim = 1)
        #x2_band_bp = torch.cat((x2_band_bp1, x2_band_bp2), dim = 1)
        #
        x1= torch.cat((x1, x1_band_bp), dim = 1)
        x2 = torch.cat((x2, x2_band_bp), dim = 1)
        
        
        #loss1 = self.instance_whitening_loss(x1, self.reversal_i, self.margin, self.num_off_diagonal)
        #loss2 = self.instance_whitening_loss(x2, self.reversal_i, self.margin, self.num_off_diagonal)
        #loss = (loss1 + loss2)/2

        x1_t = self.convx11(x1)
        x1_t = self.cstm11(x1_t)
    
        x1_t = self.convx12(x1_t) 

        x2_t = self.convx21(x2)
        x2_t = self.cstm21(x2_t)

        x2_t = self.convx22(x2_t)

        x1_t_1 = self.convseg(x1_t)
        x2_t_1 = self.convseg(x2_t)
        x_t = torch.cat((x1_t, x2_t), dim = 1)
        
        outA_f = self.dropout1(self.linears(x1_t_1.view(bs, -1))) #self.linears(x1_t_1.view(bs, -1)) #self.dropout1(self.linears(x1_t_1.view(bs, -1)))
        outB_f = self.dropout1(self.linears(x2_t_1.view(bs, -1))) #self.linears(x2_t_1.view(bs, -1)) #self.dropout1(self.linears(x2_t_1.view(bs, -1)))

        sim_all_loss, centerf_emb, seglabelA, seglabelB = self.labelsim(outA_f, outB_f, labels_A, labels_B, begin_train)
                
        out = self.linearc(self.convbn(x_t).view(bs, -1))
        
        out = self.classifier_bn(out)

        outA = self.classifier_ss(outA_f)

        outB = self.classifier_ss(outB_f)
        
        #loss =torch.tensor(0).cuda().float()
        

        return out, outA, outB, loss, outA_f, outB_f, sim_all_loss, centerf_emb, seglabelA, seglabelB


 

