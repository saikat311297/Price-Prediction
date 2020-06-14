%% ================ Part 1: Feature Normalization ================
clear ; close all; clc
fprintf('Loading data ...\n');
fprintf('Plotting data ...\n');
data = load('flatprice_training_set.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Plot the data points
scatter3(X(:,1), X(:,2), y, "filled");
figure;
plot(X(:,1), y, '*', 'MarkerSize', 7);
ylabel('Price in Lakhs');
xlabel('Area in Square feet');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);
% Add intercept term to X
X = [ones(m, 1) X];                         

% ================== Part 2: Gradient Descent =================
fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Initial Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
fprintf('Cost after runing gradientDescent: %f\n', J_history(num_iters));

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

figure;
scatter3(X(:,2), X(:,3), y, "filled");
hold on;
plot3(X(:,2), X(:,3), X*theta, '-');

% For Output
while 1
  area = input('Enter area of house for prediction (0 to exit) : ');
  if area==0
    break
  endif
  room = input('Enter Number of Rooms : ');
  temp = [1 area room];
  temp(1,2) = (temp(1,2) - mu(1,1)) / (sigma(1,1));
  temp(1,3) = (temp(1,3) - mu(1,2)) / (sigma(1,2));
  price = temp * theta;
  fprintf(['Price of a %0.2f sq-ft, %d room house ' ...
         '(using gradient descent): %0.4f lakhs (Predicted)\n'], area, room, price);
endwhile
fprintf('Program paused. Press enter to continue.\n');
pause;

Xtest = load('flatprice_training_set.txt');
fprintf('Taking some random examples from dataset predicting prices for comparison\n');
m = size(Xtest,1);
mt = 5;
price = zeros(mt,1);

rand_indices = randperm(m);
sel = Xtest(rand_indices(1:mt), :);
for i=1:mt
  temp = [1 sel(i,1) sel(i,2)];
  temp(1,2) = (temp(1,2) - mu(1,1)) / (sigma(1,1));
  temp(1,3) = (temp(1,3) - mu(1,2)) / (sigma(1,2));
  price(i) = temp * theta;
endfor

for i=1:mt
  fprintf(['Flat Size = %d, bhk = %d, Actual Price = %0.2f lakhs, Estimated price = %0.4f lakhs\n'], sel(i,1), sel(i,2), sel(i,3), price(i));
endfor