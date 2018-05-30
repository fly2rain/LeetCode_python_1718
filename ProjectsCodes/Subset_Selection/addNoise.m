function b = addNoise(varargin)
%IMNOISE Add noise to image.
%   J = addNoise (I,TYPE,...) Add noise of a given TYPE to the intensity image
%   I. TYPE is a string that can have one of these values:
%
%       'gaussian'       Gaussian white noise with constant
%                        mean and variance
%
%       'localvar'       Zero-mean Gaussian white noise
%                        with an intensity-dependent variance
%
%       'laplace'        Laplace noise with constant
%                        mean and variance
%
%       'poisson'        Poisson noise
%
%       'salt & pepper'  "On and Off" pixels
%
%       'speckle'        Multiplicative noise
%
%   Depending on TYPE, you can specify additional parameters to IMNOISE. All
%   numerical parameters are normalized; they correspond to operations with
%   images with intensities ranging from 0 to 1.
%
%   J = IMNOISE(I,'gaussian',M,V) adds Gaussian white noise of mean M and
%   variance V to the image I. When unspecified, M and V default to 0 and
%   0.01 respectively.
%
%   J = imnoise(I,'localvar',V) adds zero-mean, Gaussian white noise of
%   local variance, V, to the image I.  V is an array of the same size as I.
%
%   J = imnoise(I,'localvar',IMAGE_INTENSITY,VAR) adds zero-mean, Gaussian
%   noise to an image, I, where the local variance of the noise is a
%   function of the image intensity values in I.  IMAGE_INTENSITY and VAR
%   are vectors of the same size, and PLOT(IMAGE_INTENSITY,VAR) plots the
%   functional relationship between noise variance and image intensity.
%   IMAGE_INTENSITY must contain normalized intensity values ranging from 0
%   to 1.
%
%   J = IMNOISE(I,'laplace', M, V) adds laplace noise of mean M and
%   variance V to the image I. When unspecified, M and V default to 0 and
%   0.01 respectively.
%
%   J = IMNOISE(I,'poisson') generates Poisson noise from the data instead
%   of adding artificial noise to the data.  If I is double precision,
%   then input pixel values are interpreted as means of Poisson
%   distributions scaled up by 1e12.  For example, if an input pixel has
%   the value 5.5e-12, then the corresponding output pixel will be
%   generated from a Poisson distribution with mean of 5.5 and then scaled
%   back down by 1e12.  If I is single precision, the scale factor used is
%   1e6.  If I is uint8 or uint16, then input pixel values are used
%   directly without scaling.  For example, if a pixel in a uint8 input
%   has the value 10, then the corresponding output pixel will be
%   generated from a Poisson distribution with mean 10.
%
%   J = IMNOISE(I,'salt & pepper',D) adds "salt and pepper" noise to the
%   image I, where D is the noise density.  This affects approximately
%   D*numel(I) pixels. The default for D is 0.05.
%
%   J = IMNOISE(I,'speckle',V) adds multiplicative noise to the image I,
%   using the equation J = I + n*I, where n is uniformly distributed random
%   noise with mean 0 and variance V. The default for V is 0.04.
%
%   Note
%   ----
%   The mean and variance parameters for 'gaussian', 'localvar', and
%   'speckle' noise types are always specified as if for a double image
%   in the range [0, 1].  If the input image is of class uint8 or uint16,
%   the imnoise function converts the image to double, adds noise
%   according to the specified type and parameters, and then converts the
%   noisy image back to the same class as the input.
%
%   Class Support
%   -------------
%   For most noise types, I can be uint8, uint16, double, int16, or
%   single.  For Poisson noise, int16 is not allowed. The output
%   image J has the same class as I.  If I has more than two dimensions
%   it is treated as a multidimensional intensity image and not as an
%   RGB image.
%
%   Example
%   -------
%        I = imread('eight.tif');
%        J = imnoise(I,'salt & pepper', 0.02);
%        figure, imshow(I), figure, imshow(J)
%
%   See also RAND, RANDN.

%   Copyright 1993-2010 The MathWorks, Inc.
%   $Revision: 5.20.4.12 $  $Date: 2011/08/09 17:50:58 $

[a, code, classIn, classChanged, p3, p4] = ParseInputs(varargin{:});

clear varargin;
sizeA = size(a);

switch code
    case 'gaussian' % Gaussian white noise
        b = a + sqrt(p4)*randn(sizeA) + p3;
        
    case 'localvar_1' % Gaussian white noise with variance varying locally
        % imnoise(a,'localvar',v)
        % v is local variance array
        b = a + sqrt(p3).*randn(sizeA); % Use a local variance array
        
    case 'localvar_2' % Gaussian white noise with variance varying locally
        % Use an empirical intensity-variance relation
        intensity = p3(:); % Use an empirical intensity-variance relation
        var       = p4(:);
        minI  = min(intensity);
        maxI  = max(intensity);
        b     = min(max(a,minI),maxI);
        b     = reshape(interp1(intensity,var,b(:)),sizeA);
        b     = a + sqrt(b).*randn(sizeA);
        
    case 'laplace'
        % Generate Laplacian noise
        u = rand(sizeA)-0.5;
        b = a + p3 + sqrt(p4/2) * sign(u).* log(1- 2 * abs(u));
        
    case 'poisson' % Poisson noise
        switch classIn
            case 'uint8'
                a = round(a*255);
            case 'uint16'
                a = round(a*65535);
            case 'single'
                a = a * 1e6;  % Recalibration
            case 'double'
                a = a * 1e12; % Recalibration
        end
        
        a = a(:);
        
        %  (Monte-Carlo Rejection Method) Ref. Numerical
        %  Recipes in C, 2nd Edition, Press, Teukolsky,
        %  Vetterling, Flannery (Cambridge Press)
        
        b = zeros(size(a));
        idx1 = find(a<50); % Cases where pixel intensities are less than 50 units
        if (~isempty(idx1))
            g = exp(-a(idx1));
            em = -ones(size(g));
            t = ones(size(g));
            idx2 = (1:length(idx1))';
            while ~isempty(idx2)
                em(idx2) = em(idx2) + 1;
                t(idx2) = t(idx2) .* rand(size(idx2));
                idx2 = idx2(t(idx2) > g(idx2));
            end
            b(idx1) = em;
        end
        
        % For large pixel intensities the Poisson pdf becomes
        % very similar to a Gaussian pdf of mean and of variance
        % equal to the local pixel intensities. Ref. Mathematical Methods
        % of Physics, 2nd Edition, Mathews, Walker (Addison Wesley)
        idx1 = find(a >= 50); % Cases where pixel intensities are at least 50 units
        if (~isempty(idx1))
            b(idx1) = round(a(idx1) + sqrt(a(idx1)) .* randn(size(idx1)));
        end
        
        b = reshape(b,sizeA);
        
    case 'salt & pepper' % Salt & pepper noise
        b = a;
        x = rand(sizeA);
        b(x < p3/2) = 0; % Minimum value
        b(x >= p3/2 & x < p3) = 1; % Maximum (saturated) value
        
    case 'speckle' % Speckle (multiplicative) noise
        b = a + sqrt(12*p3)*a.*(rand(sizeA)-.5);
        
end

% Truncate the output array data if necessary
if strcmp(code,{'poisson'})
    switch classIn
        case 'uint8'
            b = uint8(b);
        case 'uint16'
            b = uint16(b);
        case 'single' % b = max(0, min(b / 1e6, 1));
            b = max(-1e6, min(b / 1e6, 1));
        case 'double' % b = max(0, min(b / 1e12, 1));
            b = max(-1e12, min(b / 1e12, 1));
    end
else
    %     b = max(0,min(b,1));
    % The output class should be the same as the input class
    if classChanged,
        b = changeclass(classIn, b);
    end
end


%%%
%%% ParseInputs
%%%
function [a, code, classIn, classChanged, p3, p4, msg] = ParseInputs(varargin)

% Initialization
p3           = [];
p4           = [];
msg = '';

% Check the number of input arguments.

narginchk(1,4);

% Check the input-array type.
a = varargin{1};
validateattributes(a, {'uint8','uint16','double','int16','single'}, {}, mfilename, ...
    'I', 1);

% Change class to double
classIn = class(a);
classChanged = 0;
if ~isa(a, 'double')
    a = double(a);
    classChanged = 1;
else
    % Clip so a is between 0 and 1.
    %     a = max(min(a,1),0);
end

% Check the noise type.
if nargin > 1
    if ~ischar(varargin{2})
        error(message('images:imnoise:invalidNoiseType'))
    end
    
    % Preprocess noise type string to detect abbreviations.
    allStrings = {'gaussian', 'salt & pepper', 'speckle',...
        'laplace', 'poisson','localvar'};
    idx = find(strncmpi(varargin{2}, allStrings, numel(varargin{2})));
    switch length(idx)
        case 0
            error(message('images:imnoise:unknownNoiseType', varargin{ 2 }))
        case 1
            code = allStrings{idx};
        otherwise
            error(message('images:imnoise:ambiguousNoiseType', varargin{ 2 }))
    end
else
    code = 'gaussian';  % default noise type
end

switch code
    case 'poisson'
        if nargin > 2
            error(message('images:imnoise:tooManyPoissonInputs'))
        end
        
        if strcmp(classIn, 'int16')
            error(message('images:imnoise:badClassForPoisson'));
        end
        
    case 'gaussian'
        p3 = 0;     % default mean
        p4 = 0.01;  % default variance
        
        if nargin > 2
            p3 = varargin{3};
            if ~isRealScalar(p3)
                error(message('images:imnoise:invalidMean'))
            end
        end
        
        if nargin > 3
            p4 = varargin{4};
            if ~isNonnegativeRealScalar(p4)
                error(message('images:imnoise:invalidVariance', 'gaussian'))
            end
        end
        
        if nargin > 4
            error(message('images:imnoise:tooManyGaussianInputs'))
        end
        
    case 'laplace'
        p3 = 0;     % default mean
        p4 = 0.01;  % default variance
        
        if nargin > 2
            p3 = varargin{3};
            if ~isRealScalar(p3)
                error(message('images:imnoise:invalidMean'))
            end
        end
        
        if nargin > 3
            p4 = varargin{4};
            if ~isNonnegativeRealScalar(p4)
                error(message('images:imnoise:invalidVariance', 'gaussian'))
            end
        end
        
        if nargin > 4
            error(message('images:imnoise:tooManyGaussianInputs'))
        end
        
    case 'salt & pepper'
        p3 = 0.05;   % default density
        
        if nargin > 2
            p3 = varargin{3};
            if ~isNonnegativeRealScalar(p3) || (p3 > 1)
                error(message('images:imnoise:invalidNoiseDensity'))
            end
            
            if nargin > 3
                error(message('images:imnoise:tooManySaltAndPepperInputs'))
            end
        end
        
    case 'speckle'
        p3 = 0.05;    % default variance
        
        if nargin > 2
            p3 = varargin{3};
            if ~isNonnegativeRealScalar(p3)
                error(message('images:imnoise:invalidVariance', 'speckle'))
            end
        end
        
        if nargin > 3
            error(message('images:imnoise:tooManySpeckleInputs'))
        end
        
    case 'localvar'
        if nargin < 3
            error(message('images:imnoise:toofewLocalVarInputs'))
            
        elseif nargin == 3
            % IMNOISE(a,'localvar',v)
            code = 'localvar_1';
            p3 = varargin{3};
            if ~isNonnegativeReal(p3) || ~isequal(size(p3),size(a))
                error(message('images:imnoise:invalidLocalVarianceValueAndSize'))
            end
            
        elseif nargin == 4
            % IMNOISE(a,'localvar',IMAGE_INTENSITY,NOISE_VARIANCE)
            code = 'localvar_2';
            p3 = varargin{3};
            p4 = varargin{4};
            
            if ~isNonnegativeRealVector(p3) || (any(p3) > 1)
                error(message('images:imnoise:invalidImageIntensity'))
            end
            
            if ~isNonnegativeRealVector(p4)
                error(message('images:imnoise:invalidLocalVariance'))
            end
            
            if ~isequal(size(p3),size(p4))
                error(message('images:imnoise:invalidSize'))
            end
            
        else
            error(message('images:imnoise:tooManyLocalVarInputs'))
        end
        
end

%%%
%%% isReal
%%%
function t = isReal(P)
%   isReal(P) returns 1 if P contains only real
%   numbers and returns 0 otherwise.
%
isFinite  = all(isfinite(P(:)));
t = isreal(P) && isFinite && ~isempty(P);


%%%
%%% isNonnegativeReal
%%%
function t = isNonnegativeReal(P)
%   isNonnegativeReal(P) returns 1 if P contains only real
%   numbers greater than or equal to 0 and returns 0 otherwise.
%
t = isReal(P) && all(P(:)>=0);


%%%
%%% isRealScalar
%%%
function t = isRealScalar(P)
%   isRealScalar(P) returns 1 if P is a real,
%   scalar number and returns 0 otherwise.
%
t = isReal(P) && (numel(P)==1);


%%%
%%% isNonnegativeRealScalar
%%%
function t = isNonnegativeRealScalar(P)
%   isNonnegativeRealScalar(P) returns 1 if P is a real,
%   scalar number greater than 0 and returns 0 otherwise.
%
t = isReal(P) && all(P(:)>=0) && (numel(P)==1);


%%%
%%% isVector
%%%
function t = isVector(P)
%   isVector(P) returns 1 if P is a vector and returns 0 otherwise.
%
t = ((numel(P) >= 2) && ((size(P,1) == 1) || (size(P,2) == 1)));


%%%
%%% isNonnegativeRealVector
%%%
function t = isNonnegativeRealVector(P)
%   isNonnegativeRealVector(P) returns 1 if P is a real,
%   vector greater than 0 and returns 0 otherwise.
%
t = isReal(P) && all(P(:)>=0) && isVector(P);
