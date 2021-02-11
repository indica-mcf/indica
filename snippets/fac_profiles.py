def ne_te(te0=3.e3, ne0=6.e19):
    """ Profiles of electron density and temperature with typical shape of a
    tokamak H-mode discharge. 
    
    Profiles are monotonically increasing towards the plasma centre, with te0 
    and ne0 are the central values at rhop = 0
    """
    
    import numpy as np
    
    temp = [1.000e+00,9.989e-01,9.955e-01,9.900e-01,9.822e-01,9.722e-01,9.602e-01, 
            9.461e-01,9.300e-01,9.121e-01,8.924e-01,8.711e-01,8.482e-01,8.240e-01,7.985e-01, 
            7.719e-01,7.444e-01,7.161e-01,6.872e-01,6.578e-01,6.280e-01,5.982e-01,5.684e-01, 
            5.388e-01,5.097e-01,4.813e-01,4.536e-01,4.267e-01,4.009e-01,3.760e-01,3.523e-01, 
            3.296e-01,3.081e-01,2.878e-01,2.686e-01,2.505e-01,2.336e-01,2.271e-01,2.208e-01, 
            2.146e-01,2.086e-01,2.028e-01,1.972e-01,1.202e-01,7.253e-02,4.421e-02,2.668e-02, 
            1.627e-02,9.816e-03,5.984e-03,3.611e-03,2.201e-03,1.328e-03,8.098e-04,4.887e-04, 
            2.979e-04,1.798e-04]
   
    dens = [1.0000000, 0.99970878, 0.99884519, 0.99742657, 0.99546973, 0.99299314, 
            0.99001570, 0.98655672, 0.98263662, 0.97827527,  0.97349394, 0.96831317, 
            0.96275470, 0.95683957, 0.95058901, 0.94402472, 0.93716776, 0.93003935, 
            0.92266030, 0.91505136, 0.90723332, 0.89922581, 0.89104855, 0.88272116, 
            0.87426213, 0.86569021, 0.85702312, 0.84827836, 0.83947271, 0.83062234, 
            0.82174359, 0.81285132, 0.80396018, 0.79508443, 0.78623769, 0.77743270,
            0.76868217, 0.76520048, 0.76172973, 0.75827062, 0.75482386, 0.75139200,
            0.74797388, 0.70906474, 0.53350715, 0.38471636, 0.27635022, 0.19763660,
            0.14063114, 0.10141023, 0.072845232, 0.052096520, 0.037070032, 0.026731381, 
            0.019201744, 0.013732453, 0.0097715192]

    rhop = [0.0000,0.0250,0.0500,0.0750,0.1000,0.1250,0.1500,0.1750, 
            0.2000,0.2250,0.2500,0.2750,0.3000,0.3250,0.3500,0.3750,
            0.4000,0.4250,0.4500,0.4750,0.5000,0.5250,0.5500,0.5750,
            0.6000,0.6250,0.6500,0.6750,0.7000,0.7250,0.7500,0.7750,
            0.8000,0.8250,0.8500,0.8750,0.9000,0.9100,0.9200,0.9300,
            0.9400,0.9500,0.9600,0.9700,0.9800,0.9900,1.0000,1.0100,
            1.0200,1.0300,1.0400,1.0500,1.0600,1.0700,1.0800,1.0900,
            1.1000]

    rhop = np.array(rhop)
    temp = np.array(temp) * te0
    dens = np.array(dens) * ne0

    return rhop, temp, dens