function seed_rand()
try
  rng(3, 'twister')
catch
  rand('twister',3);
end