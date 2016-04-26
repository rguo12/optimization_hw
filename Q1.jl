using DataFrames
function goal(x::Array{Float64,1})
  x1 = x[1]
  x2 = x[2]
  return 200*x1^2 + x2^2
end

function gradient_goal(x::Array{Float64,1})
  x1 = x[1]
  x2 = x[2]
  return [400*x1;2*x2]
end

function hessian_goal(x::Array{Float64,1})
  x1 = x[1]
  x2 = x[2]
  hes_x1_x1 = 400
  hes_x1_x2 = 0
  hes_x2_x2 = 2
  return [hes_x1_x1 hes_x1_x2;hes_x1_x2 hes_x2_x2]
end

function BTLS_gradient(x::Array{Float64,1},alpha::Float64,beta::Float64)
  step = 1.0
  grad = gradient_goal(x)
  x_prev = x
  while 1 > 0
    armijo_condition = goal(x_prev) + alpha*step*grad'*(-grad)
    x = x_prev - step*gradient_goal(x)
    if goal(x) < armijo_condition[1]
      break
    end
    step = beta * step
  end
  return step
end

function BTLS_newton(x::Array{Float64,1},alpha::Float64,beta::Float64)
  newton_step = 1.0
  grad = gradient_goal(x)
  hes = hessian_goal(x)
  newton_direction = -inv(hes)*grad
  x_prev = x
  while 1 > 0
    armijo_condition = goal(x_prev) + alpha*newton_step*grad'*newton_direction
    x = x_prev + newton_step*newton_direction
    if goal(x) < armijo_condition[1]
      break
    end
    newton_step = newton_step*beta
  end
  return newton_step
end

#parameters
alpha = 0.2
beta = 0.5
#x = squeeze(rand(1,2)*10.0-5,1)
x = [2.918089 ; -1.288549]
epsilon = 0.1^4
f_value = goal(x)
grad = gradient_goal(x)
hes = hessian_goal(x)
#m = "gradient_descent"
m = "newton_method"
iter_cnt = 0
step = 1
newton_step = 1
round_acc = 6
lambda_square = grad'*inv(hes)*grad
row = round([iter_cnt x[1] x[2] f_value norm(grad) step],round_acc)
if m == "gradient_descent"
  while norm(grad) >= epsilon
    step = BTLS_gradient(x,alpha,beta)
    x = x-step*grad
    f_value = goal(x)
    grad = gradient_goal(x)
    iter_cnt += 1
    row = vcat(row,round([iter_cnt x[1] x[2] f_value norm(grad) step],round_acc))
  end
end
if m == "newton_method"
  while lambda_square[1]/2.0 > epsilon
    #norm(grad) >= epsilon
    newton_step = BTLS_newton(x,alpha,beta)
    print(x)
    println()
    print(-newton_step*inv(hes)*grad)
    x = x-newton_step*inv(hes)*grad
    f_value = goal(x)
    grad = gradient_goal(x)
    hes = hessian_goal(x)
    lambda_square = grad'*inv(hes)*grad
    iter_cnt += 1
    row = vcat(row,round([iter_cnt x[1] x[2] f_value norm(grad) newton_step],round_acc))
  end
end
df = convert(DataFrame,row)
writetable(join(["Q1/output_alpha_",string(alpha),"_beta_",string(beta),"_method_",m,".csv"]), df, separator = ',', header = false)
