#workspace()
using DataFrames
function target_function(x::Array{Float64,1},t::Float64)
  x1 = x[1]
  x2 = x[2]
  x3 = x[3]
  #print(x)
  #println()
  t*(1/x1+1/x2+1/x3) - (log(2-x1-x2) + log(2-x1-x3) + log(2-x2-x3)) - log(x1*x2*x3)
end

function target_gradient(x::Array{Float64,1},t::Float64)
  x1 = x[1]
  x2 = x[2]
  x3 = x[3]
  grad_x1 = -t/x1^2 - (-1/(2-x1-x2)) - (-1/(2-x1-x3)) - 1/x1
  grad_x2 = -t/x2^2 - (-1/(2-x1-x2)) - (-1/(2-x2-x3)) - 1/x2
  grad_x3 = -t/x3^2 - (-1/(2-x3-x2)) - (-1/(2-x1-x3)) - 1/x1
  return [grad_x1;grad_x2;grad_x3]
end

function target_hessian(x::Array{Float64,1},t::Float64)
  x1 = x[1]
  x2 = x[2]
  x3 = x[3]
  hes_x1_x1 = 2*t/x1^3 + 1/(2-x1-x2)^2 + 1/(2-x1-x3)^2 + 1/x1^2
  hes_x1_x2 = 1/(2-x1-x2)^2
  hes_x1_x3 = 1/(2-x1-x3)^2
  hes_x2_x2 = 2*t/x2^3 + 1/(2-x1-x2)^2 + 1/(2-x2-x3)^2 + 1/x2^2
  hes_x2_x3 = 1/(2-x2-x3)^2
  hes_x3_x3 = 2*t/x3^3 + 1/(2-x1-x3)^2 + 1/(2-x2-x3)^2 + 1/x3^2
  return [hes_x1_x1 hes_x1_x2 hes_x1_x3;hes_x1_x2 hes_x2_x2 hes_x2_x3; hes_x1_x3 hes_x2_x3 hes_x3_x3]
end

function BTLS_newton_barrier(x::Array{Float64,1},alpha::Float64,beta::Float64,t::Float64,A::Array{Float64,2},b::Array{Float64,1})
  newton_step = 1/beta
  newton_direction = -inv(target_hessian(x,t))*target_gradient(x,t)
  #x_old = x
  #x_new = x
  #x_new = x + newton_step*newton_direction
  #function_value_new = target_function(x_new,t)

  while 1 > 0
    #function_value_new > armijo_condition[1]
    newton_step = beta * newton_step
    x_new = x + newton_step*newton_direction
    if minimum(b-A*x_new) < 0
      #not feasible, skip
      continue
    end
    function_value_new = target_function(x_new,t)
    armijo_condition = target_function(x,t) + alpha * newton_step * target_gradient(x,t)'* newton_direction
    if function_value_new <= armijo_condition[1]
      break
    end
  end
  return newton_step
end


#parameters
alpha = 0.01
beta = 0.8
epsilon = 0.1^5
mu = 20
t = 1.0 #barrier factor
x = [0.5;0.5;0.5]
A = [1.0 1 0;0 1 1;1 0 1;0 -1 0;0 -1 0;0 0 -1]
b = [2.0;2.0;2.0;0.0;0.0;0.0]
b_loop_cnt = 0
record_array = zeros(Float64,1,9)
while 6.0/t >= epsilon
  @printf "b loop No.%d \n" b_loop_cnt
  #centering step
  cent_cnt = 0
  #record = [0 x' 1.0/x[1]+1.0/x[2]+1.0/x[3] 1.0 t]
  grad = target_gradient(x,t)
  hes = target_hessian(x,t)
  lambda_square = (grad'*inv(hes)*grad)[1]
  #row = [0 x[1] x[2] x[3] 1.0/x[1]+1.0/x[2]+1.0/x[3] 1 norm(grad) lambda_square t]
  #record_array = vcat(record_array,row)
  while lambda_square/2.0 > epsilon
    #@printf "%dth centering loop \n" cent_cnt
    #newton_direction = - inv(target_hessian(x,t))*target_gradient(x,t)
    x_old = x
    newton_step = BTLS_newton_barrier(x_old,alpha,beta,t,A,b)
    newton_direction = -inv(target_hessian(x,t))*target_gradient(x,t)
    x = x + newton_step*newton_direction

    #=
    if cent_cnt > 0
      newton_step = BTLS_newton_barrier(x_new,alpha,beta,t,A,b)
    else
      newton_step = BTLS_newton_barrier(x,alpha,beta,t,A,b)
      x_new = x
    end
    newton_direction = -inv(target_hessian(x_new,t))*target_gradient(x_new,t)
    x_new = x_new + newton_step*newton_direction
    =#

    cent_cnt = cent_cnt + 1

    #print(new_row)
    #println()
    grad = target_gradient(x,t)
    hes = target_hessian(x,t)
    lambda_square = (grad'*inv(hes)*grad)[1]
    new_row = [t cent_cnt x[1] x[2] x[3] 1.0/x[1]+1.0/x[2]+1.0/x[3] newton_step norm(grad) lambda_square]
    round_new_row = round(new_row,5)
    record_array = vcat(record_array,round_new_row)
    @printf "lambda_square is %.6f" lambda_square
    println()
    #record = vcat(record,new_row)
  end
  t = mu * t
  b_loop_cnt += 1
end

df = convert(DataFrame,record_array)
writetable(join(["Q2/output_alpha_",string(alpha),"_beta_",string(beta),"_mu_",string(mu),".csv"]), df, separator = ',', header = false)
