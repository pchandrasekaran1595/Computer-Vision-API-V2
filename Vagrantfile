Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/focal64"
  config.vm.box_version = "~> "
  config.vm.boot_timeout = 900
  
  config.vm.provider "virtualbox" do |v|
    v.memory = 1024
    v.cpus = 2
  end
  
  config.vm.network "forwarded_ports", guest: 4040, host: 4040
  
  config.vm.provision "shell", inline: <<-SHELL
    systemctl disable apt-daily.service
    systemctl disable apt-daily.timer
    
    sudo apt-get update -y
    sudo apt-get install python3-venv python3-opencv zip -y
  SHELL
end
