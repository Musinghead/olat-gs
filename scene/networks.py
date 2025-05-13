import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from utils.general_utils import dot_prod


class ScatterNet(nn.Module):
    """
    input: lat code per gaussian point, light direction, view direction
    output: scattering
    """
    def __init__(self,
                 d_in,
                 d_latcode,
                 d_h,
                 d_out,
                 encoding_cfg):
        super(ScatterNet, self).__init__()
        self.encoding = tcnn.Encoding(d_in, encoding_cfg, dtype=torch.float32)
        self.network = nn.Sequential(
            nn.Linear(self.encoding.n_output_dims * 2 + d_latcode + d_in * 3 + 1, d_h),
            nn.ReLU(True),
            nn.Linear(d_h, d_h),
            nn.ReLU(True),
            nn.Linear(d_h, d_h),
            nn.ReLU(True),
            nn.Linear(d_h, d_h),
            nn.ReLU(True),
            nn.Linear(d_h, d_out)
        ).cuda()

    def forward(self, wi, wo, latcodes, normal, spec_cue):
        """

        Args:
            wi: n x 3
            latcodes: n x c

        Returns:

        """

        wi_enc = self.encoding((wi + 1.) / 2.)
        wo_enc = self.encoding((wo + 1.) / 2.)

        x = torch.cat([wi_enc, wo_enc, latcodes, wi, wo, normal, spec_cue], dim=-1)
        out = torch.exp(self.network(x))
        return out


class IncidentNet(nn.Module):
    """
    input: lat code per gaussian point, light direction
    output: light incident factor
    """
    def __init__(self,
                 d_in,
                 d_latcode,
                 d_h,
                 d_out,
                 encoding_cfg):
        super(IncidentNet, self).__init__()
        self.encoding = tcnn.Encoding(d_in, encoding_cfg, dtype=torch.float32)
        self.network = nn.Sequential(
            nn.Linear(self.encoding.n_output_dims + d_latcode + d_in + 1, d_h),
            nn.ReLU(True),
            nn.Linear(d_h, d_h),
            nn.ReLU(True),
            nn.Linear(d_h, d_h),
            nn.ReLU(True),
            nn.Linear(d_h, d_h),
            nn.ReLU(True),
            nn.Linear(d_h, d_out)
        ).cuda()

    def forward(self, wi, latcodes, visi_cue, foreshorten_cue):
        """

        Args:
            wi: n x 3, unit vector
            latcodes: n x c

        Returns:

        """

        wi_enc = self.encoding((wi+1.)/2.)

        x = torch.cat([wi_enc, latcodes, wi, visi_cue * foreshorten_cue], dim=-1)
        out = torch.sigmoid(self.network(x))
        return out
