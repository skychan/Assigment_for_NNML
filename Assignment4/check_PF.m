## Copyright (C) 2017 Sky Chan
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} new (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Sky Chan <sky@Chen.local>
## Created: 2017-10-03
function [log_z] = new(rbm_w)
%Calculates Z inefficient. Not for a matrix of 10x256
%If rbm_w parameter equal zero, generate matrix of 10x3 to test
if (rbm_w == 0)
    A = magic(20);
    rbm_w = A(1:10,1:3)/max(max(A));
end

matrix_dim = size(rbm_w);
number_of_hidden_nodes = matrix_dim(1);
number_of_visible_nodes = matrix_dim(2);
n=number_of_hidden_nodes + number_of_visible_nodes;
B = generate_all_binary_states(n);
visible_state = B(:,1:number_of_visible_nodes)';
hidden_state = B(:,number_of_visible_nodes+1:number_of_visible_nodes+number_of_hidden_nodes)';

G = diag(visible_state'*rbm_w'*hidden_state);
E = exp(G);
Z = sum(E);
P = E / Z;
log_z = log(Z);
end

function B = generate_all_binary_states(n)
% B returns all binary inputs
N = 2^n;
D = (0:N-1)';
B = rem(floor(D*pow2(-(n-1):0)),2);
end
